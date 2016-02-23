"""
Metrics for analyzing networks, etc.
"""
from __future__ import division, print_function
from itertools import chain
from more_itertools import unique_everseen
import numpy as np


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


def softmax_prob_from_weights(weights, gain):
    """
    Convert a weight matrix to a probabilistic transition matrix using a softmax rule.
    :param weights: weight matrix (rows are targs, cols are srcs)
    :param gain: scaling factor in softmax function (higher gain -> lower entropy)
    :return: transition matrix (rows are targs, cols are srcs), initial probabilities
    """

    # calculate transition probabilities p
    p_unnormed = np.exp(gain * weights)
    p = p_unnormed / p_unnormed.sum(0)

    # define p0 to be stationary distribution, given by principal eigenvector
    evals, evecs = np.linalg.eig(p)
    idxs_sorted = np.real(evals).argsort()[::-1]

    # normalize
    p0_unnormed = evecs[:, idxs_sorted[0]]
    p0 = p0_unnormed / p0_unnormed.sum()

    return p, p0


def path_probability(path, p, p0):
    """
    Calculate the probability of a given path through a softmax network.
    :param path: sequence of nodes
    :param p: transition matrix (NOTE: rows are targs, cols are srcs)
    :param p0: initial node probability
    :return: probability of path
    """

    prob = p0[path[0]]
    for node_ctr, node in enumerate(path[1:]):
        prob *= p[node, path[node_ctr]]

    return prob


def most_probable_paths(weights, gain, length, n):
    """
    Find the most probable paths through a network when node transition probabilities
    are given via a softmax function.

    :param weights: weight matrix (rows are targs, cols are srcs)
    :param gain: scaling factor in softmax function
    :param length: length of paths to look for
    :param n: number of paths to return
    :return: list of n most probable paths through network
    """

    paths = paths_of_length(weights, length)

    # get transition matrix
    p, p0 = softmax_prob_from_weights(weights, gain)

    # calculate path probabilities
    probs = np.zeros((len(paths),), dtype=float)

    for path_ctr, path in enumerate(paths):
        probs[path_ctr] = path_probability(path, p, p0)

    # sort calculated probabilities
    idxs_sorted = probs.argsort()[::-1]

    # return paths corresponding to n highest probabilities
    paths_array = np.array(paths)[idxs_sorted[:n]]

    # return paths as list of tuples
    return [tuple(path) for path in paths_array]


def reorder_by_paths(x, paths):
    """
    Reorder the columns activity matrix according to a set of
    the most probable paths through a network. Useful for visualizing repeated sequences.
    :param x: activity matrix (rows are timepoints, cols are nodes)
    :param paths: list of tuple paths
    :return: x with optimally permuted columns
    """

    # find all the nodes that appear in probable paths, ordered by those paths
    ordering = list(unique_everseen(chain.from_iterable(paths)))

    # add in the rest of the nodes randomly
    for node in range(x.shape[1]):
        if node not in ordering:
            ordering.append(node)

    # reorder the columns
    return x[:, np.array(ordering)], ordering