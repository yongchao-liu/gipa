import random

import dgl
import numpy as np
import torch as th
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator


def load_dataset(dataset_type, **kwargs):
    """
    Load dataset.
    Args:
        dataset_type: str, support 'proteins', 'cora', 'citeseer', 'pubmed', 'amazon', 'reddit'.
    """
    if dataset_type == 'proteins':
        data = DglNodePropPredDataset(name='ogbn-proteins', root=kwargs['root'])
        evaluator = Evaluator(name='ogbn-proteins')

        splitted_idx = data.get_idx_split()
        train_idx, val_idx, test_idx = splitted_idx["train"], splitted_idx["valid"], splitted_idx["test"]
        graph, labels = data[0]
        species = graph.ndata['species']
        features = one_hot_encoder(species)
        graph.ndata['feat'] = features
        graph.ndata['label'] = labels

        return graph, labels, train_idx, val_idx, test_idx, evaluator

    if dataset_type == 'cora':
        dataset = dgl.data.CoraGraphDataset()
    elif dataset_type == 'citeseer':
        dataset = dgl.data.CiteseerGraphDataset()
    elif dataset_type == 'pubmed':
        dataset = dgl.data.PubmedGraphDataset()
    elif dataset_type == 'amazon':
        dataset = dgl.data.AmazonCoBuyComputerDataset()
    elif dataset_type == 'reddit':
        dataset = dgl.data.RedditDataset()
    else:
        raise (KeyError('Dataset type {} not recognized.'.format(dataset_type)))

    if dataset_type == 'amazon':
        num_classes = dataset.num_classes
        graph = dataset[0]
        features = th.FloatTensor(graph.ndata['feat'])
        labels = th.LongTensor(graph.ndata['label'])
    else:
        num_classes = dataset.num_classes
        graph = dataset[0]
        features = th.FloatTensor(graph.ndata['feat'])
        labels = th.LongTensor(graph.ndata['label'])
        train_mask = th.BoolTensor(graph.ndata['train_mask'])
        val_mask = th.BoolTensor(graph.ndata['val_mask'])
        test_mask = th.BoolTensor(graph.ndata['test_mask'])

    return graph, features, labels, num_classes, train_mask, val_mask, test_mask


def one_hot_encoder(x):
    """
    Get ont hot embedding of the input tensor.
    Args:
        x: torch.Tensor, input 1-D tensor.
    Returns:
        one_hot: torch.Tensor, one-hot embedding of x.
    """
    ids = x.unique()
    id_dict = dict(list(zip(ids.numpy(), np.arange(len(ids)))))
    one_hot = th.zeros((len(x), len(ids)))
    for i, u in enumerate(x):
        if id_dict[u.item()] == 4:
            pass
        else:
            one_hot[i][id_dict[u.item()]] = 1

    return one_hot


def evaluate(model, features, labels, mask):
    """
    Evaluate accuracy of the input model.
    Args:
        model: torch.nn.module.
        features: torch.Tensor.
        labels: torch.LongTensor.
        mask: torch.Tensor.
    Returns:
        average accuracy.
    """
    model.eval()
    with th.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = th.max(logits, dim=1)
        correct = th.sum(indices == labels)

        return correct.item() * 1.0 / len(labels)


class DataLoaderWrapper(object):
    """
    Wrapper of dgl.dataloading.NodeDataLoader.
    """
    def __init__(self, dataloader):
        self.iter = iter(dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.iter)
        except Exception:
            raise StopIteration() from None


class BatchSampler(object):
    """
    Batch sampler for dgl.dataloading.NodeDataLoader.
    """
    def __init__(self, n_node, batch_size):
        self.n_node = n_node
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            n_shuffle = th.randperm(self.n_node).split(self.batch_size)
            for batch in n_shuffle:
                yield batch
            yield None


def seed(seed=0):
    """
    Fix random process by a seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
    dgl.random.seed(seed)


def get_aggregator(agg_type):
    """
    Get the aggregator function.
    Args:
        agg_type: str, 'sum', 'max', 'mean'.
    Returns:
        dgl.function
    """
    if agg_type == 'sum':
        return dgl.function.sum
    elif agg_type == 'max':
        return dgl.function.max
    elif agg_type == 'mean':
        return dgl.function.mean
    else:
        raise KeyError('Aggregator type {} not recognized.'.format(agg_type))


def get_activation(act_type):
    """
    Get the activation function.
    Args:
        act_type: str, 'relu', 'leaky_relu'.
    Returns:
        torch.nn.functional
    """
    if act_type == 'relu':
        return th.nn.functional.relu
    elif act_type == 'leaky_relu':
        return th.nn.functional.leaky_relu
    else:
        raise KeyError('Activation type {} not recognized.'.format(act_type))
