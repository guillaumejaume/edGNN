import torch
import torch.nn as nn
import dgl.function as fn

from core.models.constants import GNN_EDGE_LABELS_KEY, GNN_EDGE_NORM, GNN_MSG_KEY, GNN_NODE_FEAT_IN_KEY


class RGCNLayer(nn.Module):
    def __init__(self, g, node_dim, edge_dim, out_feats, activation=None, dropout=None, bias=None):
        """

        :param g: DGLGraph
        :param node_dim:
        :param edge_dim:
        :param out_feats:
        :param activation:
        :param dropout:
        :param bias:
        """

        super(RGCNLayer, self).__init__()

        # 1. set attributes
        self.bias = bias
        self.activation = activation
        self.in_feat = node_dim
        self.out_feat = out_feats
        self.num_rels = edge_dim
        self.is_input_layer = False
        self.num_bases = self.num_rels

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        # 2. create variables
        if self.bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
            nn.init.xavier_uniform_(self.bias,
                                    gain=nn.init.calculate_gain('relu'))

        self.weight = nn.Parameter(torch.Tensor(self.num_bases, self.in_feat,
                                                self.out_feat))
        if self.num_bases < self.num_rels:
            # linear combination coefficients
            self.w_comp = nn.Parameter(torch.Tensor(self.num_rels,
                                                    self.num_bases))

        # 3. initialize variables
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
        if self.num_bases < self.num_rels:
            nn.init.xavier_uniform_(self.w_comp,
                                    gain=nn.init.calculate_gain('relu'))

    def propagate(self, g):

        def msg_func(edges):
            w = self.weight[edges.data[GNN_EDGE_LABELS_KEY]]
            msg = torch.bmm(edges.src[GNN_NODE_FEAT_IN_KEY].unsqueeze(1), w).squeeze()
            msg = msg * edges.data[GNN_EDGE_NORM].unsqueeze(1)
            return {GNN_MSG_KEY: msg}
        g.update_all(msg_func, fn.sum(msg=GNN_MSG_KEY, out=GNN_NODE_FEAT_IN_KEY), None)

    def forward(self, node_features, edge_features, g):

        g.ndata[GNN_NODE_FEAT_IN_KEY] = node_features

        self.propagate(g)

        # apply bias and activation
        node_repr = g.ndata[GNN_NODE_FEAT_IN_KEY]
        if self.bias:
            node_repr = node_repr + self.bias
        if self.activation:
            node_repr = self.activation(node_repr)

        return node_repr

