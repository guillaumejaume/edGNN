import torch
import numpy as np
from distutils.util import strtobool

from dgl.contrib.data import load_data
from dgl import DGLGraph

from core.data.constants import GRAPH, LABELS, TRAIN_MASK, TEST_MASK, VAL_MASK, N_RELS, N_CLASSES
from core.models.constants import GNN_EDGE_LABELS_KEY, GNN_EDGE_NORM
from core.data.utils import complete_path, load_pickle, save_pickle


def preprocess_dglrgcn(*, dataset, out_folder, bfs_level=3, relabel=False, reverse_edges=False):
    """

    :param dataset:
    :param out_folder:
    :param bfs_level:
    :param relabel:
    :param reverse_edges: backwards edges are added to the graph, if True 2x more edges + 2x more num_rels
    :return:
    """
    if isinstance(bfs_level, str):
        bfs_level = int(bfs_level)
    if isinstance(relabel, str):
        relabel = strtobool(relabel)
    if isinstance(reverse_edges, str):
        reverse_edges = strtobool(reverse_edges)
      
    data = load_data(dataset=dataset, bfs_level=bfs_level, relabel=relabel)

    labels = torch.squeeze(torch.LongTensor(data.labels))

    def _idx_to_mask(idx, n):
        mask = np.zeros(n, dtype=int)
        mask[idx] = 1
        return torch.ByteTensor(mask)

    val_idx = data.train_idx[:len(data.train_idx) // 5]
    val_mask = _idx_to_mask(val_idx, labels.shape[0])

    train_idx = data.train_idx[len(data.train_idx) // 5:]
    train_mask = _idx_to_mask(train_idx, labels.shape[0])

    test_mask = _idx_to_mask(data.test_idx, labels.shape[0])

    n_rels = data.num_rels

    # graph preprocess and calculate normalization factor
    g = DGLGraph()

    g.add_nodes(data.num_nodes)

    edge_src, edge_dst, edge_type = data.edge_src, data.edge_dst, torch.LongTensor(data.edge_type)

    if reverse_edges:
        g.add_edges(edge_src, edge_dst)
        g.add_edges(edge_dst, edge_src)
        edge_type = torch.cat((edge_type, edge_type + n_rels), 0)
        g.edata[GNN_EDGE_LABELS_KEY] = edge_type
    else:
        g.add_edges(edge_src, edge_dst)
        g.edata[GNN_EDGE_LABELS_KEY] = edge_type
        g.edata[GNN_EDGE_NORM] = torch.from_numpy(data.edge_norm).unsqueeze(1)

    save_pickle(g, complete_path(out_folder, GRAPH))
    save_pickle(2*n_rels if reverse_edges else n_rels, complete_path(out_folder, N_RELS))
    save_pickle(data.num_classes, complete_path(out_folder, N_CLASSES))
    torch.save(labels, complete_path(out_folder, LABELS))
    torch.save(train_mask, complete_path(out_folder, TRAIN_MASK))
    torch.save(test_mask, complete_path(out_folder, TEST_MASK))
    torch.save(val_mask, complete_path(out_folder, VAL_MASK))


def load_dglrgcn(folder):
    data = {
        GRAPH: load_pickle(complete_path(folder, GRAPH)),
        N_RELS: load_pickle(complete_path(folder, N_RELS)),
        N_CLASSES: load_pickle(complete_path(folder, N_CLASSES))
    }

    for k in [LABELS, TRAIN_MASK, TEST_MASK, VAL_MASK]:
        data[k] = torch.load(complete_path(folder, k))

    return data
