import argparse
import os
import time

import dgl
import dgl.function as fn
import numpy as np
import torch as th
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import gipa_model
import utils


def preprocess(graph, use_label=False):
    # add additional features
    graph.update_all(fn.copy_e("feat", "e"), fn.sum("e", "feat_add"))
    if use_label:
        graph.ndata['feat'] = th.cat((graph.ndata['feat_add'], graph.ndata['feat']), dim=1)
    else:
        graph.ndata['feat'] = graph.ndata['feat_add']
    graph.create_formats_()

    return graph


def rocauc(pred, labels, evaluator):
    return evaluator.eval({"y_pred": pred, "y_true": labels})["rocauc"]


@th.no_grad()
def evaluate(model, dataloader, labels, train_idx, val_idx, test_idx, criterion, evaluator):
    model.eval()
    preds = th.zeros(labels.shape).to(dev)
    for input_nodes, output_nodes, subgraphs in dataloader:
        subgraphs = [b.to(dev) for b in subgraphs]
        pred = model(subgraphs)
        preds[output_nodes] += pred
        # th.cuda.empty_cache()

    train_score = rocauc(preds[train_idx], labels[train_idx], evaluator)
    val_score = rocauc(preds[val_idx], labels[val_idx], evaluator)
    test_score = rocauc(preds[test_idx], labels[test_idx], evaluator)
    train_loss = criterion(preds[train_idx], labels[train_idx].float()).item()
    val_loss = criterion(preds[val_idx], labels[val_idx].float()).item()
    test_loss = criterion(preds[test_idx], labels[test_idx].float()).item()

    return train_score, val_score, test_score, train_loss, val_loss, test_loss


@th.no_grad()
def evaluate_test(model, graph, labels, test_idx, evaluator):
    model.eval()
    model.cpu()
    preds = model(graph)
    model.cuda()

    return rocauc(preds[test_idx], labels[test_idx], evaluator)


def train(args):
    # load dataset
    graph, labels, train_idx, val_idx, test_idx, evaluator = utils.load_dataset(args.dataset)
    if args.preprocess:
        graph = preprocess(graph, args.use_label)
    labels = labels.to(dev)
    train_idx = train_idx.to(dev)
    val_idx = val_idx.to(dev)
    test_idx = test_idx.to(dev)
    n_classes = labels.shape[1]
    n_node_feat = graph.ndata["feat"].shape[-1]
    n_edge_feat = graph.edata["feat"].shape[-1]

    # add self loop (should be false for proteins data)
    if args.self_loop:
        graph = dgl.add_self_loop(graph)

    # normalization
    if args.normalize:
        degs = graph.in_degrees().float()
        norm = th.pow(degs, -0.5)
        norm[th.isinf(norm)] = 0
        norm = norm.to(dev)
        graph.ndata['norm'] = norm.unsqueeze(1)

    # create GIPA model
    model = gipa_model.GIPA(n_node_feat=n_node_feat,
                            n_edge_feat=n_edge_feat,
                            n_node_emb=args.n_node_emb,
                            n_edge_emb=args.n_edge_emb,
                            n_hiddens_att=args.n_hiddens_att,
                            n_heads_att=args.n_heads_att,
                            n_hiddens_prop=args.n_hiddens_prop,
                            n_hiddens_agg=args.n_hiddens_agg,
                            n_hiddens_deep=args.n_hiddens_deep,
                            n_layers=args.n_layers,
                            n_classes=n_classes,
                            agg_type=args.agg_type,
                            act_type=args.act_type,
                            edge_drop=args.edge_drop,
                            dropout_node=args.dropout_node,
                            dropout_att=args.dropout_att,
                            dropout_prop=args.dropout_prop,
                            dropout_agg=args.dropout_agg,
                            dropout_deep=args.dropout_deep)

    print("Number of parameters:", sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad]))

    model = model.to(dev)
    loss_fcn = nn.BCEWithLogitsLoss()
    optimizer = th.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.75, patience=50,
                                                           verbose=True)
    if args.sampling > 0:
        train_sampler = dgl.dataloading.MultiLayerNeighborSampler([args.sampling for _ in range(args.n_layers)])
    else:
        train_sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args.n_layers)

    train_dataloader = utils.DataLoaderWrapper(
        dgl.dataloading.NodeDataLoader(
            graph.cpu(),
            train_idx.cpu(),
            train_sampler,
            batch_sampler=utils.BatchSampler(len(train_idx), batch_size=args.batch_size),
            num_workers=8,
        )
    )

    if args.sampling > 0:
        eval_sampler = dgl.dataloading.MultiLayerNeighborSampler([args.sampling for _ in range(args.n_layers)])
    else:
        eval_sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args.n_layers)

    eval_dataloader = utils.DataLoaderWrapper(
        dgl.dataloading.NodeDataLoader(
            graph.cpu(),
            th.cat([train_idx.cpu(), val_idx.cpu(), test_idx.cpu()]),
            eval_sampler,
            batch_sampler=utils.BatchSampler(graph.number_of_nodes(), batch_size=args.batch_size),
            num_workers=8,
        )
    )

    # tensorboard monitor
    model_name = "gipa_{}_{}".format(args.dataset, cur_time)
    LOG_PATH = "./log/{}".format(model_name)
    if not os.path.exists(LOG_PATH):
        os.mkdir(LOG_PATH)
    writer = SummaryWriter(LOG_PATH)

    # mini-batch training
    dur = []
    best_val_score = 0.0
    final_test_score = 0.0
    for epoch in range(args.n_epochs):
        model.train()
        t0 = time.time()
        for input_nodes, output_nodes, blocks in train_dataloader:
            subgraphs = [b.to(dev) for b in blocks]
            batch_ids = th.arange(len(output_nodes))
            batch_loss = loss_fcn(model(subgraphs)[batch_ids], subgraphs[-1].dstdata['label'][batch_ids].float())
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            # th.cuda.empty_cache()

        if epoch % args.eval_every == 0:
            train_score, val_score, test_score, train_loss, val_loss, test_loss = evaluate(
                model, eval_dataloader, labels, train_idx, val_idx, test_idx, loss_fcn, evaluator)

            if val_score > best_val_score:
                best_val_score = val_score
                final_test_score = test_score
                if final_test_score > 0.85:
                    th.save(model.state_dict(),
                            './saved_models/gipa_proteins_auc_{:.4f}_checkpoint.pt'.format(final_test_score))
                    print("Checkpoint saved.")

            lr_scheduler.step(val_score)
            writer.add_scalar(model_name + '/1_train_loss', train_loss, epoch)
            writer.add_scalar(model_name + '/2_val_loss', val_loss, epoch)
            writer.add_scalar(model_name + '/3_test_loss', test_loss, epoch)
            writer.add_scalar(model_name + '/4_train_auc', train_score, epoch)
            writer.add_scalar(model_name + '/5_val_auc', val_score, epoch)
            writer.add_scalar(model_name + '/6_test_auc', test_score, epoch)
            dur.append(time.time() - t0)

            print("Epoch {:05d} | Time(s) {:.4f} | Train Loss {:.4f} | Train AUC {:.4f} | Val AUC {:.4f} | "
                  "Test AUC {:.4f}".format(epoch, np.mean(dur), train_loss, train_score, val_score, test_score))

        if epoch % args.eval_best_every == 0 and epoch != 0:
            print("Best Val AUC {:.2%}, Test AUC {:.2%}".format(best_val_score, final_test_score))
            if final_test_score > 0.85:
                full_test_score = evaluate_test(model, graph, labels, test_idx, evaluator)
                print("Full test AUC {:.2%}".format(full_test_score))

    print("gipa_{}".format(cur_time))
    print("Best Val AUC {:.2%}".format(best_val_score))
    print("Final Test AUC {:.2%}".format(final_test_score))
    if args.if_save:
        th.save(model.state_dict(), './saved_models/gipa_proteins_auc_{:.4f}.pt'.format(final_test_score))
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GIPA')
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--dataset", type=str, default='proteins',
                        help="dataset: cora, citeseer, pubmed, amazon, reddit, proteins")
    parser.add_argument("--seed", type=int, default=0, help="random seed")

    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--n-layers", type=int, default=6, help="number of gipa layers")
    parser.add_argument("--n-epochs", type=int, default=7000, help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1000, help="mini-batch size")
    parser.add_argument("--sampling", type=int, default=16, help="sampling size")
    parser.add_argument("--n-hop", type=int, default=1, help="number of hops")
    parser.add_argument("--weight-decay", type=float, default=0, help="Weight for L2 loss")
    parser.add_argument("--n-node-emb", type=int, default=80, help="size of node feature embedding")
    parser.add_argument("--dropout-node", type=float, default=0.1, help="dropout probability of node features")
    parser.add_argument("--n-edge-emb", type=int, default=16, help="size of edge feature embedding")
    parser.add_argument("--edge-drop", type=float, default=0.1, help="dropout probability of edge features")

    parser.add_argument("--n-hiddens-att", type=list, default=[80],
                        help="list of number of attention hidden units")
    parser.add_argument("--n-heads-att", type=int, default=8, help="number of attention heads")
    parser.add_argument("--dropout-att", type=float, default=0.1, help="dropout probability of attention layers")
    parser.add_argument("--n-hiddens-prop", type=list, default=[80],
                        help="list of number of propagation hidden units")
    parser.add_argument("--dropout-prop", type=float, default=0.25, help="dropout probability of propagation layers")
    parser.add_argument("--n-hiddens-agg", type=list, default=[],
                        help="list of number of aggregation hidden units")
    parser.add_argument("--dropout-agg", type=float, default=0.25, help="dropout probability of aggregation layers")
    parser.add_argument("--n-hiddens-deep", type=list, default=[], help="list of number of deep hidden units")
    parser.add_argument("--dropout-deep", type=float, default=0.5, help="dropout probability of deep layers")
    parser.add_argument("--eval-every", type=int, default=5, help="evaluation frequency")
    parser.add_argument("--eval-best-every", type=int, default=20, help="best evaluation frequency")

    parser.add_argument("--agg-type", type=str, default='sum', help="aggregation type")
    parser.add_argument("--act-type", type=str, default='relu', help="activation type")
    parser.add_argument("--self-loop", action='store_true', help="graph self-loop (default=False)")
    parser.add_argument("--normalize", action='store_true', help="graph normalization (default=False)")
    parser.add_argument("--preprocess", action='store_true', help="graph preprocessing (default=False)")
    parser.add_argument("--use-label", action='store_true', help="use label as node features (default=False)")
    parser.add_argument("--if-save", action='store_true', help="save the best model (default=False)")

    args = parser.parse_args()

    global dev, cur_time
    if args.gpu < 0:
        dev = th.device('cpu')
    else:
        dev = th.device('cuda:{}'.format(args.gpu))

    utils.seed(args.seed)

    # saving configuration
    cur_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    argsDict = args.__dict__
    with open("./config/config_gipa_{}_{}.txt".format(args.dataset, cur_time), 'w') as f:
        for arg, value in argsDict.items():
            f.writelines("{}: {}\n".format(arg, value))

    print("gipa_{}".format(cur_time))
    print(args)
    train(args)
