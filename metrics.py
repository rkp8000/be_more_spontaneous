"""
Metrics for analyzing networks, etc.
"""
from __future__ import division, print_function


def paths_of_length(weights, length, start=None):
    """
    Find all paths of a given length emanating from a starting node.
    :param weights: weight matrix (rows are targs, cols are srcs)
    :param length: how many nodes to look down the tree
    :param start: what node to start from
    :return: list of all paths of specified length from starting node
    """

    if start:
        paths = [(start,)]
    else:
        paths = [(node,) for node in range(weights.shape[0])]

    for depth in range(length):

        new_paths = []

        for path in paths:
            new_paths += [path + (node,) for node in weights[:, path[-1]].nonzero()[0]]

        paths = new_paths

    return paths