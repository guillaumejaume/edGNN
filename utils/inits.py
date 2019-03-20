import math
import torch
from torch.nn import Linear
import torch.nn as nn
from dgl import DGLGraph


def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)


def init_weights(m):
    if isinstance(m, Linear):
        stdv = 1. / math.sqrt(m.weight.size(1))
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.uniform_(-stdv, stdv)


def to_cuda(data):
    print('*** Set variables on CUDA ***')

    def _set_graph_on_cuda(graph):
        for key_graph, val_graph in graph.ndata.items():
            graph.ndata[key_graph] = graph.ndata.pop(key_graph).cuda()
        for key_graph, val_graph in graph.edata.items():
            graph.edata[key_graph] = graph.edata.pop(key_graph).cuda()

    for key, val in data.items():
        if isinstance(val, torch.Tensor):
            data[key] = val.cuda()
        if isinstance(val, DGLGraph):
            _set_graph_on_cuda(val)
        if isinstance(val, list):
            for datapt in val:
                if isinstance(datapt, DGLGraph):
                    _set_graph_on_cuda(datapt)
    return data
