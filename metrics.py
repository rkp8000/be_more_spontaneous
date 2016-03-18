"""
Metrics for analyzing networks, etc.
"""
from __future__ import division, print_function
from copy import deepcopy
from itertools import chain
from more_itertools import unique_everseen
import numpy as np
from scipy import stats


def paths_of_length(weights, length, start=None):
    """
    Find all paths of a given length emanating from a starting node.
    :param weights: weight matrix (rows are targs, cols are srcs)
    :param length: how many nodes to look down the tree
    :param start: what node to start from
    :return: list of all paths of specified length from starting node
    """

    if start is not None:
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
    return [tuple(path) for path in paths_array], np.array(sorted(probs))[::-1]


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


def first_node_pair_non_overlapping_path_tree(nodes, weights, length, allow_path_loops=True):
    """
    Given a sorted list of nodes of a certain length through a network, find the first pair of
    path-starting nodes whose emanating path trees are completely non overlapping
    :param nodes: sorted list of nodes to check
    :param weights: weight matrix (rows are targs, cols are srcs)
    :param length: path length
    :param allow_path_loops: whether to allow a node to be returned if its paths include loops
    :return: tuple containing two nodes
    """

    # get all paths emanating from node_0
    for node_0 in nodes:
        node_0_path_tree = paths_of_length(weights, length, start=node_0)
        node_0_path_tree_elements = set(np.array(node_0_path_tree).flatten())

        if not node_0_path_tree:
            continue

        # loop through
        for node_1 in nodes[node_0 + 1:]:
            node_1_path_tree = paths_of_length(weights, length, start=node_1)
            node_1_path_tree_elements = set(np.array(node_1_path_tree).flatten())

            if not node_1_path_tree:
                continue

            # save this node if there is no overlap between their path trees
            if not node_0_path_tree_elements & node_1_path_tree_elements:
                if not allow_path_loops:
                    if any([len(set(path)) != len(path) for path in node_0_path_tree]):
                        continue
                    if any([len(set(path)) != len(path) for path in node_1_path_tree]):
                        continue
                return node_0, node_1

    else:
        return None


def path_tree_overlaps(nodes, weights, length):
    """
    Find the path tree overlap for every pair of nodes for paths of a specific length.

    :param nodes: which nodes to look at
    :param weights: full weight matrix (rows are targs, cols are srcs)
    :param length: number of edges in paths
    :return: overlap matrix, list of path trees for specified nodes
    """

    path_trees = [
        list(chain(*[paths_of_length(weights, length, start=node) for length in range(1, length + 1)]))
        for node in nodes
    ]

    #path_trees = [paths_of_length(weights, length, start=node) for node in nodes]

    overlap = np.zeros((len(nodes), len(nodes)), dtype=int)

    for node_0_ctr, node_0 in enumerate(nodes):
        for node_1_ctr, node_1 in enumerate(nodes):

            node_0_elements = set(chain(*path_trees[node_0_ctr]))
            node_1_elements = set(chain(*path_trees[node_1_ctr]))

            overlap[node_0_ctr, node_1_ctr] = len(node_0_elements & node_1_elements)

    return overlap, path_trees


def paths_include_loops(paths):
    """
    Check if there are any repeated elements in any of the paths in a list of paths.
    :param paths: list of paths
    :return: True if any path contains repeated nodes.
    """

    return any([len(set(path)) != len(path) for path in paths])


def get_number_of_past_occurrences_of_most_recent_sequence(seq, l):
    """
    Calculate how many times the most recent l-length sequence has occurred in the past.
    :param seq: sequence
    :param l: length of recent sequence to look for
    :return: scalar indicating how many times most recent sequence has occurred
    """

    occs = np.zeros((len(seq) - l,), dtype=bool)

    for t in range(len(seq) - l):
        if np.all(seq[-l:] == seq[t:t + l]):
            occs[t] = True

    # don't count last subsequence
    occs[-1] = False

    return occs.sum()


def get_number_of_past_occurrences(seq, l):
    """
    For every time point in the sequence, calculate how many times the most recent l-length sequence has occurred
    in the past.
    :param seq: integer sequence
    :param l: length of sequence to look for
    :return: array of same length as sequence
    """

    occs = np.zeros(seq.shape, dtype=int)

    for t in range(l + 1, len(seq) + 1):

        occs[t-1] = get_number_of_past_occurrences_of_most_recent_sequence(seq[:t], l)

    return occs


def get_number_of_past_occurrences_of_specific_sequence(seq, subseq):
    """
    Calculate how many times in the past a specific subsequence has occurred, for every time in large sequence.
    :param seq: large sequence
    :param subseq: subsequence to look for
    :return: increasing array of same length as sequence
    """

    occs = np.zeros(seq.shape, dtype=int)

    for t in range(len(subseq), len(seq) + 1):
        if np.all(seq[t - len(subseq):t] == subseq):
            occs[t - 1] = 1

    return np.cumsum(occs)


def spontaneous_sequence_repeats_stats(ntwk, ls, n_steps, n_repeats):
    """
    Estimate the expected number of times the most recent sequence has spontaneously occurred
    in the past t time steps for t < n_steps.

    :param ntwk: binary network
    :param ls: lengths of sequences to investigate
    :param n_steps: how many steps to run network for to look for repeats
    :param n_repeats: how many times to repeat the spontaneous run to make the estimate
    :return: means, stds, and sems
    """
    ntwk_base = ntwk

    past_occurrences = {l: [] for l in ls}

    for _ in range(n_repeats):

        ntwk = deepcopy(ntwk_base)
        ntwk.store_voltages = True

        for _ in range(n_steps):
            ntwk.step()

        spikes = np.array(ntwk.rs_history)
        sequence = spikes.nonzero()[1]

        for l in ls:
            past_occurrences[l].append(get_number_of_past_occurrences(sequence, l))

    means = [np.mean(np.array(past_occurrences[l]), axis=0) for l in ls]
    stds = [np.std(np.array(past_occurrences[l]), axis=0) for l in ls]
    sems = [stats.sem(np.array(past_occurrences[l]), axis=0) for l in ls]

    return means, stds, sems