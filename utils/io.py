import torch
import os
import json
import uuid
import dgl


def print_graph_stats(g, name='Graph'):
    if isinstance(g, dgl.DGLGraph):
        print("""\n----Graph statistics------'
          Name %s
          # Edges %d
          # Nodes %d""" %
              (name,
               g.number_of_edges(),
               g.number_of_nodes()))


def save_checkpoint(model, save_path=''):
    """
    Save a checkpoint model.

    """
    torch.save(model.state_dict(), os.path.join(save_path))


def load_checkpoint(model, load_path=''):
    model.load_state_dict(torch.load(load_path))
    return model


def read_params(fname, verbose=False):
    """
        Config file contains either a simple config set or a list of configs
        (used to run several experiments).
    :param fname:
    :param reading_index:
    :param verbose:
    :return: config params
    """
    # @TODO proper json schema checker
    with open(fname, 'r') as in_config:
        data = json.load(in_config)
        if 'configs' in data:
            config_params = data['configs']
        else:
            config_params = [data]
        if verbose:
            print('\n*** Model config parameters:', config_params)
    return config_params


def save_results(result, path='', verbose=False):
    if not path:
        path = '../results.json'  # default destination is root project folder
    if verbose:
        print("Save result entry in ", path)
    # 1) Load existing file if it exists or create if it doesn't
    if os.path.isfile(path):
        with open(path, 'r') as in_file:
            results = json.load(in_file)
    else:
        results = {'results': []}
    # 2) Append result
    results['results'].append(result)
    # 3) Resave results
    with open(path, 'w') as outfile:
        json.dump(results, outfile, indent=4)


def create_default_path():
    path = 'checkpoint_' + str(uuid.uuid4()) + '.pt'
    return path


def remove_model(file):
    os.remove(file)
