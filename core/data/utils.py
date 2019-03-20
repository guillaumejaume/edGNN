import os
import dill as pickle


def save_pickle(obj, ofpath):
    """
    Save an object as pickle

    Args:
        graph (DGLGraph): graph to be saved
        ofpath (str): path where to store the file
    """
    with open(ofpath, 'wb') as ofh:
        pickle.dump(obj, ofh)


def load_pickle(ifpath):
    """
    Load an object from pickle

    Args:
        ifpath (str): path from where a graph is loaded
    """
    with open(ifpath, 'rb') as ifh:
        return pickle.load(ifh)


def complete_path(folder, fname):
    return os.path.join(folder, fname)
