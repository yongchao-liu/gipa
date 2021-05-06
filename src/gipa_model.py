import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.utils import expand_as_pair

import utils


class GIPAConv(nn.Module):
    def __init__(self,
                 n_node_feat,
                 n_edge_feat,
                 n_hiddens_att,
                 n_heads_att,
                 n_hiddens_prop,
                 n_hiddens_agg,
                 agg_type,
                 act_type,
                 edge_drop=0,
                 dropout_att=0,
                 dropout_prop=0,
                 dropout_agg=0):
        """
        GIPA convolution.
        Args:
            n_node_feat: int, number of input node features.
            n_edge_feat: int, number of input edge features.
            n_hiddens_att: list of int, number of hidden units in each attention layer.
            n_heads_att: int, number of head in multi-head attention.
            n_hiddens_prop: list of int, number of hidden units in each propagation layer.
            n_hiddens_agg: list of int, number of hidden units in each aggregation layer.
            agg_type: str, 'sum', 'max', 'mean'.
            act_type: srt, 'relu', 'leaky_relu'.
            edge_drop: float, rate for edge drop.
            dropout_att: float, dropout rate for attention layer.
            dropout_prop: float, dropout rate for propagation layer.
            dropout_agg: float, dropout rate for aggregation layer.
        """
        super(GIPAConv, self).__init__()
        self._n_hiddens_att = n_hiddens_att
        self._n_heads_att = n_heads_att
        self._n_hiddens_prop = n_hiddens_prop
        self._n_hiddens_agg = n_hiddens_agg

        # attention layers
        self.att_src_layers = nn.ModuleList()
        self.att_dst_layers = nn.ModuleList()
        self.att_edge_layers = nn.ModuleList()
        if len(n_hiddens_att) == 0:
            self.att_src_layers.append(nn.Linear(n_node_feat, n_heads_att, bias=False))
            self.att_dst_layers.append(nn.Linear(n_node_feat, n_heads_att, bias=False))
            self.att_edge_layers.append(nn.Linear(n_edge_feat, n_heads_att, bias=False))
        else:
            for i in range(len(n_hiddens_att)):
                if i == 0:
                    self.att_src_layers.append(nn.Linear(n_node_feat, n_hiddens_att[0], bias=False))
                    self.att_dst_layers.append(nn.Linear(n_node_feat, n_hiddens_att[0], bias=False))
                    self.att_edge_layers.append(nn.Linear(n_edge_feat, n_hiddens_att[0], bias=False))
                else:
                    self.att_src_layers.append(nn.Linear(n_hiddens_att[i - 1], n_hiddens_att[i], bias=False))
                    self.att_dst_layers.append(nn.Linear(n_hiddens_att[i - 1], n_hiddens_att[i], bias=False))
                    self.att_edge_layers.append(nn.Linear(n_hiddens_att[i - 1], n_hiddens_att[i], bias=False))
            self.att_src_layers.append(nn.Linear(n_hiddens_att[-1], n_heads_att, bias=False))
            self.att_dst_layers.append(nn.Linear(n_hiddens_att[-1], n_heads_att, bias=False))
            self.att_edge_layers.append(nn.Linear(n_hiddens_att[-1], n_heads_att, bias=False))

        # propagation layers
        self.src_prop_layers = nn.ModuleList()
        self.dst_prop_layers = nn.ModuleList()
        if len(n_hiddens_prop) == 1:
            self.src_prop_layers.append(nn.Linear(n_node_feat, n_hiddens_prop[-1] * n_heads_att))
            self.dst_prop_layers.append(nn.Linear(n_node_feat, n_hiddens_prop[-1] * n_heads_att))
        else:
            for i in range(len(n_hiddens_prop)):
                if i == 0:
                    self.src_prop_layers.append(nn.Linear(n_node_feat, n_hiddens_prop[i]))
                    self.dst_prop_layers.append(nn.Linear(n_node_feat, n_hiddens_prop[i]))
                elif i != len(n_hiddens_prop) - 1:
                    self.src_prop_layers.append(nn.Linear(n_hiddens_prop[i - 1], n_hiddens_prop[i]))
                    self.dst_prop_layers.append(nn.Linear(n_hiddens_prop[i - 1], n_hiddens_prop[i]))
                else:
                    self.src_prop_layers.append(nn.Linear(n_hiddens_prop[i - 1], n_hiddens_prop[-1] * n_heads_att))
                    self.dst_prop_layers.append(nn.Linear(n_hiddens_prop[i - 1], n_hiddens_prop[-1] * n_heads_att))

        # aggregation layers
        self.agg_layers = nn.ModuleList()
        for i in range(len(n_hiddens_agg)):
            if i == 0:
                self.agg_layers.append(nn.Linear(2 * n_hiddens_prop[-1] * n_heads_att, n_hiddens_agg[i]))
            else:
                self.agg_layers.append(nn.Linear(n_hiddens_agg[i - 1], n_hiddens_agg[i]))

        self.edge_drop = edge_drop
        self.dropout_att = nn.Dropout(p=dropout_att)
        self.dropout_prop = nn.Dropout(p=dropout_prop)
        self.dropout_agg = nn.Dropout(p=dropout_agg)

        self.reducer = utils.get_aggregator(agg_type)
        self.activation = utils.get_activation(act_type)
        self.reset_parameters(act_type)

    def reset_parameters(self, act_type):
        """
        Initialize parameters in each layer.
        """
        if act_type == 'relu':
            gain = nn.init.calculate_gain('relu')
        elif act_type == 'leaky_relu':
            gain = nn.init.calculate_gain('leaky_relu', 0.2)
        else:
            gain = nn.init.calculate_gain('relu')

        # propagation initialization
        for i in range(len(self.src_prop_layers)):
            nn.init.xavier_normal_(self.src_prop_layers[i].weight, gain=gain)
            nn.init.xavier_normal_(self.dst_prop_layers[i].weight, gain=gain)

        # attention initialization
        for i in range(len(self.att_edge_layers)):
            nn.init.xavier_normal_(self.att_edge_layers[i].weight, gain=gain)
        for i in range(len(self.att_src_layers)):
            nn.init.xavier_normal_(self.att_src_layers[i].weight, gain=gain)
        for i in range(len(self.att_dst_layers)):
            nn.init.xavier_normal_(self.att_dst_layers[i].weight, gain=gain)

        # aggregation initialization
        for i in range(len(self.agg_layers)):
            nn.init.xavier_normal_(self.agg_layers[i].weight, gain=gain)

    def forward(self, graph, feat, feat_edge):
        """
        Forward process of GIPA convolution.
        Args:
            graph: dgl.DGLgraph, graph with topology.
            feat: torch.Tensor, features of nodes.
            feat_edge: torch.Tensor, features of edges.
        Returns:
            rst: torch.Tensor, result tensor.
        """
        with graph.local_scope():
            feat_src, feat_dst = expand_as_pair(feat, graph)

            # src and dst propagation
            prop_src = feat_src
            prop_dst = feat_dst
            for i in range(len(self._n_hiddens_prop)):
                prop_src = self.src_prop_layers[i](prop_src)
                prop_dst = self.dst_prop_layers[i](prop_dst)
                if i != len(self._n_hiddens_prop) - 1:
                    prop_src = self.activation(prop_src)
                    prop_src = self.dropout_prop(prop_src)
                    prop_dst = self.activation(prop_dst)
                    prop_dst = self.dropout_prop(prop_dst)
            prop_src = prop_src.view(-1, self._n_heads_att, self._n_hiddens_prop[-1])
            prop_dst = prop_dst.view(-1, self._n_heads_att, self._n_hiddens_prop[-1])

            # multi-head attention
            att_src = feat_src
            att_dst = feat_dst
            att_edge = feat_edge
            for i in range(len(self.att_src_layers)):
                att_src = self.att_src_layers[i](att_src)
                att_dst = self.att_dst_layers[i](att_dst)
                att_edge = self.att_edge_layers[i](att_edge)
                if i != len(self.att_src_layers) - 1:
                    att_src = self.activation(att_src, inplace=True)
                    att_src = self.dropout_att(att_src)
                    att_dst = self.activation(att_dst, inplace=True)
                    att_dst = self.dropout_att(att_dst)
                    att_edge = self.activation(att_edge, inplace=True)
                    att_edge = self.dropout_att(att_edge)
            att_src = att_src.view(-1, self._n_heads_att, 1)
            att_dst = att_dst.view(-1, self._n_heads_att, 1)
            att_edge = att_edge.view(-1, self._n_heads_att, 1)

            graph.srcdata["prop_src"] = prop_src
            graph.srcdata["att_src"] = att_src
            graph.dstdata["att_dst"] = att_dst
            graph.apply_edges(fn.u_add_v("att_src", "att_dst", "att_node"))
            att = graph.edata["att_node"] + att_edge
            att = self.activation(att)

            # edge drop and edge-wise softmax
            if self.training and self.edge_drop > 0:
                perm = th.randperm(graph.number_of_edges(), device=feat.device)
                eids = perm[int(graph.number_of_edges() * self.edge_drop):]
                graph.edata["att"] = th.zeros_like(att)
                graph.edata["att"][eids] = dgl.ops.edge_softmax(graph, att[eids], eids=eids)
            else:
                graph.edata["att"] = dgl.ops.edge_softmax(graph, att)

            # message passing
            # message function update attention results and save them on edges
            # reducer function gathers information from edges
            graph.update_all(fn.u_mul_e("prop_src", "att", "m"), self.reducer('m', 'prop_src'))
            rst = graph.dstdata["prop_src"]

            # aggregation
            if len(self._n_hiddens_agg) != 0:
                rst = th.cat((prop_dst, rst), dim=1)
                for i in range(len(self.agg_layers)):
                    rst = self.agg_layers[i](rst)
                    if i != len(self.agg_layers) - 1:
                        rst = self.activation(rst, inplace=True)
                        rst = self.dropout_agg(rst)
            else:
                # direct residual connection
                rst += prop_dst

            return rst


class GIPA(nn.Module):
    def __init__(self,
                 n_node_feat,
                 n_edge_feat,
                 n_node_emb,
                 n_edge_emb,
                 n_hiddens_att,
                 n_heads_att,
                 n_hiddens_prop,
                 n_hiddens_agg,
                 n_hiddens_deep,
                 n_layers,
                 n_classes,
                 agg_type,
                 act_type,
                 edge_drop=0,
                 dropout_node=0,
                 dropout_att=0,
                 dropout_prop=0,
                 dropout_agg=0,
                 dropout_deep=0):
        """
        GIPA model.
        Args:
            n_node_feat: int, number of input node features.
            n_edge_feat: int, number of input edge features.
            n_node_emb: int, number of node embedding features.
            n_edge_emb: int, number of edge embedding features.
            n_hiddens_att: list of int, number of hidden units in each attention layer.
            n_heads_att: int, number of head in multi-head attention.
            n_hiddens_prop: list of int, number of hidden units in each propagation layer.
            n_hiddens_agg: list of int, number of hidden units in each aggregation layer.
            n_hiddens_deep: list of int, number of hidden units in each deep layer.
            n_layers: int, number of GIPA layers.
            n_classes: int, number of label classes.
            agg_type: str, 'sum', 'max', 'mean'.
            act_type: srt, 'relu', 'leaky_relu'.
            edge_drop: float, rate for edge drop.
            dropout_node: float, dropout rate for node embedding layer.
            dropout_att: float, dropout rate for attention layer.
            dropout_prop: float, dropout rate for propagation layer.
            dropout_agg: float, dropout rate for aggregation layer.
            dropout_deep: float, dropout rate for deep layer.
        """
        super(GIPA, self).__init__()
        self._n_layers = n_layers

        # node feature embedding
        if n_node_emb > 0:
            self.node_emb = nn.Linear(n_node_feat, n_node_emb, bias=False)
            n_node_feat = n_node_emb
        else:
            self.node_emb = None

        # GIPA convolution layers
        self.gipa_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.edge_embs = nn.ModuleList()
        for i in range(n_layers):
            if i == 0:
                in_feat = n_node_feat
                if len(n_hiddens_agg) == 0:
                    out_feat = n_heads_att * n_hiddens_prop[-1]
                else:
                    out_feat = n_hiddens_agg[-1]
            elif len(n_hiddens_agg) == 0:
                in_feat = n_heads_att * n_hiddens_prop[-1]
                out_feat = n_heads_att * n_hiddens_prop[-1]
            else:
                in_feat = n_hiddens_agg[-1]
                out_feat = n_hiddens_agg[-1]

            self.gipa_layers.append(
                GIPAConv(n_node_feat=in_feat,
                         n_edge_feat=n_edge_emb,
                         n_hiddens_att=n_hiddens_att,
                         n_heads_att=n_heads_att,
                         n_hiddens_prop=n_hiddens_prop,
                         n_hiddens_agg=n_hiddens_agg,
                         agg_type=agg_type,
                         act_type=act_type,
                         edge_drop=edge_drop,
                         dropout_att=dropout_att,
                         dropout_prop=dropout_prop,
                         dropout_agg=dropout_agg))
            self.edge_embs.append(nn.Linear(n_edge_feat, n_edge_emb))
            self.batch_norms.append(nn.BatchNorm1d(out_feat))

        # deep layers
        self.deep_layers = nn.ModuleList()
        if len(n_hiddens_deep) == 0:
            self.deep_layers.append(nn.Linear(out_feat, n_classes))
        else:
            for i in range(len(n_hiddens_deep)):
                if i == 0:
                    self.deep_layers.append(nn.Linear(out_feat, n_hiddens_deep[0]))
                else:
                    self.deep_layers.append(nn.Linear(n_hiddens_deep[i - 1], n_hiddens_deep[i]))
            self.deep_layers.append(nn.Linear(n_hiddens_deep[-1], n_classes))

        self.activation = utils.get_activation(act_type)
        self.dropout_node = nn.Dropout(p=dropout_node)
        self.dropout_deep = nn.Dropout(p=dropout_deep)

    def forward(self, graph):
        """
        Forward process of entire GIPA model.
        Args:
            graph: dgl.DGLgraph, graph with topology.
        Returns:
            h: torch.Tensor, final output.
        """
        if not isinstance(graph, list):
            subgraphs = [graph] * self._n_layers
        else:
            subgraphs = graph

        h = subgraphs[0].srcdata['feat']
        if self.node_emb is not None:
            h = self.node_emb(h)
            h = F.relu(h, inplace=True)
            h = self.dropout_node(h)

        # gipa graph convolution (residual connection for multi-layers)
        h_last = None
        for i in range(len(self.gipa_layers)):
            feat_edge = subgraphs[i].edata["feat"]
            feat_edge_emb = self.edge_embs[i](feat_edge)
            feat_edge_emb = F.relu(feat_edge_emb, inplace=True)
            h = self.gipa_layers[i](subgraphs[i], h, feat_edge_emb).flatten(1, -1)
            if h_last is not None:
                h += h_last[:h.shape[0], :]
            h_last = h
            h = self.batch_norms[i](h)
            h = self.activation(h, inplace=True)
            h = self.dropout_deep(h)

        # deep layers
        for i in range(len(self.deep_layers)):
            h = self.deep_layers[i](h)
            if i != len(self.deep_layers) - 1:
                h = self.activation(h, inplace=True)
                h = self.dropout_deep(h)

        return h