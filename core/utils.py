import torch
from core.models.constants import GNN_MSG_KEY, GNN_NODE_FEAT_IN_KEY, GNN_NODE_FEAT_OUT_KEY, GNN_EDGE_FEAT_KEY, GNN_AGG_MSG_KEY


def reset_graph_features(g):
    keys = [GNN_NODE_FEAT_IN_KEY, GNN_AGG_MSG_KEY, GNN_MSG_KEY, GNN_NODE_FEAT_OUT_KEY]
    for key in keys:
        if key in g.ndata:
            del g.ndata[key]
    if GNN_EDGE_FEAT_KEY in g.edata:
        del g.edata[GNN_EDGE_FEAT_KEY] 


def compute_node_degrees(g):
    """
    Given a graph, compute the degree of each node
    :param g: DGL graph
    :return: node_degrees: a tensor with the degree of each node
             node_degrees_ids: a labeled version of node_degrees (usable for 1-hot encoding)
    """
    fc = lambda i: g.in_degrees(i).item()
    node_degrees = list(map(fc, range(g.number_of_nodes())))
    unique_deg = list(set(node_degrees))
    mapping = dict(zip(unique_deg, list(range(len(unique_deg)))))
    node_degree_ids = [mapping[deg] for deg in node_degrees]
    return torch.LongTensor(node_degrees), torch.LongTensor(node_degree_ids)

