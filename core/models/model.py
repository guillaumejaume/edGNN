"""
Model Interface
"""
import copy
import torch
import importlib
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl import DGLGraph
from core.utils import compute_node_degrees
from core.models.constants import NODE_CLASSIFICATION, GRAPH_CLASSIFICATION, GNN_EDGE_LABELS_KEY, GNN_NODE_LABELS_KEY

MODULE = "core.models.layers.{}"
LAYER_MODULES = {
    'edGNNLayer': 'edgnn',
    'RGCNLayer': 'rgcn'
}

ACTIVATIONS = {
    'relu': F.relu
}


def layer_build_args(node_dim, edge_dim, n_classes, layer_params, mode):
    """
    Generator of layer arguments
    Args:
        layer_params (dict): Refer to constructor
    """
    if isinstance(layer_params['n_units'], list):
        for v in layer_params.values():
            assert isinstance(v, list), "Expected list because n_units is specified as list!"
            assert len(v) == len(layer_params['n_units']), "Expected same number of elements in lists!"
        params = copy.deepcopy(layer_params)
        n_layers = len(layer_params['n_units'])
    else:
        params = dict()
        n_layers = layer_params['n_hidden_layers']
        for k, v in layer_params.items():
            if k != 'n_hidden_layers':
                params[k] = [layer_params[k]]*n_layers

    n_units = params.pop('n_units')
    activations = params.pop('activation')
    kwargs = [dict(zip(params, t)) for t in zip(*params.values())]
    if len(kwargs) == 0:
        kwargs = [{}]*n_layers

    if n_layers == 0:
        yield node_dim, edge_dim, n_classes, None, {k: None for k in params.keys()}
    else:
        # input layer
        yield node_dim, edge_dim, n_units[0], ACTIVATIONS[activations[0]], kwargs[0]

        # hidden layers
        for i in range(n_layers - 1):
            yield n_units[i], edge_dim, n_units[i+1], ACTIVATIONS[activations[i+1]], kwargs[i]

        if mode == NODE_CLASSIFICATION:
            # output layer
            yield n_units[-1], edge_dim, n_classes, None, kwargs[-1]


class Model(nn.Module):

    def __init__(self, g, config_params, n_classes=None, n_rels=None, n_entities=None, is_cuda=False, mode=NODE_CLASSIFICATION):
        """
        Instantiate a graph neural network.

        Args:
            g (DGLGraph): a preprocessed DGLGraph
            config_json (str): path to a configuration JSON file. It must contain the following fields: 
                               "layer_type", and "layer_params". 
                               The "layer_params" should be a (nested) dictionary containing at least the fields 
                               "n_units" and "activation". "layer_params" should contain other fields that corresponds
                               to keyword arguments of the concrete layers (refer to the layers implementation).
                               The name of these additional fields should be the same as the keyword args names.
                               The parameters in "layer_params" should either be lists with the same number of elements,
                               or single values. If single values are specified, then a "n_hidden_layers" (integer) 
                               field is expected.
                               The fields "n_input" and "n_classes" are required if not specified 
        """
        super(Model, self).__init__()

        self.is_cuda = is_cuda
        self.mode = mode
        self.config_params = config_params
        self.n_rels = n_rels
        self.n_classes = n_classes
        self.n_entities = n_entities
        self.g = g

        layer_type = config_params["layer_type"]

        module = importlib.import_module(MODULE.format(LAYER_MODULES[layer_type]))
        self.Layer = getattr(module, layer_type)

        self.build_model()

    def build_model(self):

        # Build NN
        self.layers = nn.ModuleList()
        layer_params = self.config_params['layer_params']

        # Edge embeddings
        if 'edge_dim' in self.config_params:
            self.edge_dim = self.config_params['edge_dim']
            self.embed_edges = nn.Embedding(self.n_rels, self.edge_dim)
        elif 'edge_one_hot' in self.config_params:
            self.edge_dim = self.n_rels
            self.embed_edges = torch.eye(self.edge_dim, self.edge_dim)
            if self.is_cuda:
                self.embed_edges = self.embed_edges.cuda()
        else:
            self.edge_dim = None
            self.embed_edges = None

        # Node embeddings
        if self.mode == NODE_CLASSIFICATION:
            deg, deg_ids = compute_node_degrees(self.g)
            n_node_degs = torch.max(deg_ids) + 1
            self.g.ndata[GNN_NODE_LABELS_KEY] = deg_ids.cuda() if self.is_cuda else deg_ids

        if 'node_dim' in self.config_params:
            self.node_dim = self.config_params['node_dim']
            if self.n_entities is None:
                self.embed_nodes = nn.Embedding(n_node_degs, self.node_dim)
            else:
                self.embed_nodes = nn.Embedding(self.n_entities, self.node_dim)
        elif 'node_one_hot' in self.config_params:
            if self.n_entities is None:
                self.embed_nodes = torch.eye(n_node_degs, n_node_degs)
                self.node_dim = n_node_degs
            else:
                self.embed_nodes = torch.eye(self.n_entities, self.n_entities)
                self.node_dim = self.n_entities
        else:
            if isinstance(self.g, DGLGraph):  # ie, we are doing node classification
                self.node_dim = self.g.number_of_nodes()
            else:
                raise RuntimeError
            self.embed_nodes = None

        # basic tests
        assert (self.n_classes is not None)

        # build and append layers
        print('\n*** Building model ***')
        for node_dim, edge_dim, n_out, act, kwargs in layer_build_args(self.node_dim, self.edge_dim, self.n_classes,
                                                                       layer_params, self.mode):
            print('* Building new layer with args:', node_dim, edge_dim, n_out, act, kwargs)
            self.layers.append(self.Layer(self.g, node_dim, edge_dim, n_out, act, **kwargs))
        print('*** Model successfully built ***\n')

        # build readout function if graph classification
        if self.mode == GRAPH_CLASSIFICATION:
            n_hidden = layer_params['n_units']
            n_hidden = n_hidden[-1] if isinstance(n_hidden, list) else n_hidden
            self.readout = nn.Linear(layer_params['n_hidden_layers'] * n_hidden, self.n_classes)

    def forward(self, g):

        if g is not None:
            g.set_n_initializer(dgl.init.zero_initializer)
            g.set_e_initializer(dgl.init.zero_initializer)
            self.g = g

        # 1. Build node features
        if isinstance(self.embed_nodes, nn.Embedding):
            node_features = self.embed_nodes(self.g.ndata[GNN_NODE_LABELS_KEY])
        elif isinstance(self.embed_nodes, torch.Tensor):
            node_features = self.embed_nodes[self.g.ndata[GNN_NODE_LABELS_KEY]]
        else:
            node_features = torch.zeros(self.g.number_of_nodes(), self.node_dim)
        node_features = node_features.cuda() if self.is_cuda else node_features

        # 2. Build edge features
        if isinstance(self.embed_edges, nn.Embedding):
            edge_features = self.embed_edges(self.g.edata[GNN_EDGE_LABELS_KEY])
        elif isinstance(self.embed_edges, torch.Tensor):
            edge_features = self.embed_edges[self.g.edata[GNN_EDGE_LABELS_KEY]]
        else:
            edge_features = None

        # 3. Iterate over each layer
        for layer_idx, layer in enumerate(self.layers):
            if layer_idx == 0:
                h = layer(node_features, edge_features, self.g)
                self.g.ndata['h_0'] = h
            else:
                h = layer(h, edge_features, self.g)
                key = 'h_' + str(layer_idx)
                self.g.ndata[key] = h

        # 4.1 If node classification, return node embeddings
        if self.mode == NODE_CLASSIFICATION:
            return h

        # 4.2 If graph classification, construct readout function
        h_g = torch.Tensor().cuda() if self.is_cuda else torch.Tensor()
        for i in range(len(self.layers)):
            key = 'h_' + str(i)
            h_g = torch.cat((h_g, dgl.sum_nodes(g, key)), dim=1)
        h_g = self.readout(h_g)
        return h_g

    def eval_node_classification(self, labels, mask):
        self.eval()
        loss_fcn = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            logits = self(None)
            logits = logits[mask]
            labels = labels[mask]
            loss = loss_fcn(logits, labels)
            _, indices = torch.max(logits, dim=1)
            correct = torch.sum(indices == labels)
            return correct.item() * 1.0 / len(labels), loss

    def eval_graph_classification(self, labels, testing_graphs):
        self.eval()
        loss_fcn = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            logits = self(testing_graphs)
            loss = loss_fcn(logits, labels)
            _, indices = torch.max(logits, dim=1)
            correct = torch.sum(indices == labels)
            return correct.item() * 1.0 / len(labels), loss

