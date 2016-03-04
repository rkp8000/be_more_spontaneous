"""
Demonstrate some of the basic properties of a random network with dynamics
governed by the softmax rule.
"""
from __future__ import division, print_function
from copy import deepcopy
from itertools import chain
import matplotlib.pyplot as plt
import numpy as np
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.append('/Users/rkp/Dropbox/Repositories/be_more_spontaneous')


import fancy_raster
import metrics
import network


def make_and_save_network(config):
    """
    Make and save a random network we can use for later experiments. The whole point is to ensure that
    the network we've created has a few features that will make visualizing things much easier in the future. Since
    the network is random and directed, it will have many paths throughout it exhibiting a tree-like structure.
    For the purposes of our analysis, we would like to be able to focus on two distinct path trees of a certain
    maximum path length, and which include no loops. This code generates random networks until this is the case
    (which it is with a high probability, given the right settings).
    """

    SEED = config['SEED']

    N_NODES = config['N_NODES']
    P_CXN = config['P_CXN']

    W_STRONG = config['W_STRONG']

    GAIN = config['GAIN']
    REFRACTORY_STRENGTH = config['REFRACTORY_STRENGTH']

    PATH_LENGTH = config['PATH_LENGTH']
    TRIAL_LENGTH = config['TRIAL_LENGTH']
    MAX_ATTEMPTS = config['MAX_ATTEMPTS']

    SAVE_FILE_NAME = config['SAVE_FILE_NAME']

    np.random.seed(SEED)

    # loop over attempts to make network with desired properties
    for trial_ctr in range(MAX_ATTEMPTS):

        print('Trial number {}'.format(trial_ctr))

        # make a random weight matrix
        nodes = range(N_NODES)
        weights = W_STRONG * (np.random.rand(N_NODES, N_NODES) < P_CXN).astype(float)
        np.fill_diagonal(weights, 0)

        # calculate overlap of all node pairs' path trees
        path_tree_overlaps, path_trees = metrics.path_tree_overlaps(nodes, weights, PATH_LENGTH)

        # set all elements of path_tree_overlaps to -1 if one node in pair has no paths with PATH_LENGTH transitions
        path_sets_max_length = [
            [path for path in path_tree if len(path) - 1 == PATH_LENGTH]
            for path_tree in path_trees
        ]
        path_tree_sizes = np.array([len(path_set) for path_set in path_sets_max_length])
        path_tree_overlaps[path_tree_sizes == 0, :] = -1
        path_tree_overlaps[:, path_tree_sizes == 0] = -1

        # set all elements of path_tree_overlaps to -1 if one node in pair loops in its paths
        path_tree_loops = np.array([metrics.paths_include_loops(path_tree) for path_tree in path_trees])
        path_tree_overlaps[path_tree_loops, :] = -1
        path_tree_overlaps[:, path_tree_loops] = -1

        # get all node pairs with distinct but non-zero path trees
        node_pairs_with_distinct_path_trees = zip(*(path_tree_overlaps == 0).nonzero())

        # make sure there is at least one pair of nodes with distinct path trees
        if not node_pairs_with_distinct_path_trees:
            print('No node pairs with distinct, non-zero, non-looping path trees.')
            continue

        # calculate sum total of path tree sizes for every pair of nodes
        path_tree_sizes_paired = []
        for node_0, node_1 in node_pairs_with_distinct_path_trees:
            path_tree_sizes_paired.append(path_tree_sizes[node_0] + path_tree_sizes[node_1])

        path_tree_sizes_paired = np.array(path_tree_sizes_paired)

        if len(path_tree_sizes_paired[path_tree_sizes_paired > 2]) == 0:
            print('No distinct path tree pairs with summed size > 2.')
            continue

        # find node pairs with minimum summed input greater than 2
        min_size = path_tree_sizes_paired[path_tree_sizes_paired > 2].min()
        print('Min summed path tree size: {}.'.format(min_size))
        candidate_node_pair_idxs = np.where(path_tree_sizes_paired == min_size)[0]
        candidate_node_pairs = np.array(node_pairs_with_distinct_path_trees)[candidate_node_pair_idxs]

        # if there are multiple node pairs with same minimum, pick pair with highest probabilities
        candidate_probabilities = []
        _, p0 = metrics.softmax_prob_from_weights(weights, GAIN)
        for node_0, node_1 in candidate_node_pairs:
            candidate_probabilities.append(p0[node_0] * p0[node_1])
        candidate_probabilities = np.array(candidate_probabilities)

        # choose final starting nodes for all subsequent examples
        node_0, node_1 = candidate_node_pairs[candidate_probabilities.argmax()]
        # get their path trees
        node_0_path_tree = list(chain(*[
            metrics.paths_of_length(weights, length, start=node_0) for length in range(1, TRIAL_LENGTH + 1)
        ]))
        node_1_path_tree = list(chain(*[
            metrics.paths_of_length(weights, length, start=node_1) for length in range(1, TRIAL_LENGTH + 1)
        ]))

        # swap if necessary so that node_0_path_tree has at least 2 paths
        if len(node_0_path_tree) == 1:
            node_0, node_1 = node_1, node_0
            node_0_path_tree, node_1_path_tree = node_1_path_tree, node_0_path_tree

        print('node_0 before reordering: {}'.format(node_0))
        print('node_1 before reordering: {}'.format(node_1))
        print('node_0_path_tree before reordering')
        print(node_0_path_tree)
        print('node_1_path_tree before reordering')
        print(node_1_path_tree)

        # relabel everything for optimal sequential visualization
        _, reordering = metrics.reorder_by_paths(weights, node_0_path_tree + node_1_path_tree)
        node_0 = reordering.index(node_0)
        node_1 = reordering.index(node_1)

        # relabel weight matrix rows and columns
        weights = weights[reordering, :]
        weights = weights[:, reordering]

        print('reordering:')
        print(reordering)

        break

    else:
        print('Maximum number of trials exceeded!')
        return

    # make a network with this weight matrix
    ntwk = network.RecurrentSoftMaxLingeringModel(
        weights, GAIN, REFRACTORY_STRENGTH, lingering_input_value=0, lingering_input_timescale=0,
    )

    # bind two root nodes and path trees to network
    ntwk.node_0 = node_0
    ntwk.node_1 = node_1
    ntwk.node_0_path_tree = metrics.paths_of_length(weights, PATH_LENGTH, node_0)
    ntwk.node_1_path_tree = metrics.paths_of_length(weights, PATH_LENGTH, node_1)

    print('node_0 after reordering: {}'.format(node_0))
    print('node_1 after reordering: {}'.format(node_1))
    print('node_0_path_tree after reordering')
    print(ntwk.node_0_path_tree)
    print('node_1_path_tree after reordering')
    print(ntwk.node_1_path_tree)

    # save the network
    np.save(SAVE_FILE_NAME, [ntwk])


def spontaneous(config):
    """
    Run simulation of spontaneous activity.
    """

    SEED = config['SEED']

    LOAD_FILE_NAME = config['LOAD_FILE_NAME']

    RUN_LENGTH_NO_DRIVE = config['RUN_LENGTH_NO_DRIVE']
    DRIVE_AMPLITUDE = config['DRIVE_AMPLITUDE']
    TRIAL_LENGTH = config['TRIAL_LENGTH']
    N_REPEATS = config['N_REPEATS']

    FIG_SIZE = config['FIG_SIZE']

    np.random.seed(SEED)

    ntwk_base = np.load(LOAD_FILE_NAME)[0]

    # let the network run spontaneously
    ntwk = deepcopy(ntwk_base)
    ntwk.store_voltages = True

    for _ in range(RUN_LENGTH_NO_DRIVE):
        ntwk.step()

    spikes = np.array(ntwk.rs_history)

    fig, axs = plt.subplots(3, 1, figsize=FIG_SIZE, tight_layout=True)

    axs[0].set_title('Free-running spontaneous activity')
    rand_perm = np.random.permutation(range(spikes.shape[1]))
    fancy_raster.by_row_circles(axs[0], spikes[:, rand_perm], drives=None)

    axs[0].set_xlim(-1, RUN_LENGTH_NO_DRIVE)
    axs[0].set_ylim(-1, ntwk.w.shape[0])
    axs[0].set_xlabel('time step')
    axs[0].set_ylabel('ensemble')

    axs[1].set_title('Free-running spontaneous activity (partially resorted)')
    fancy_raster.by_row_circles(axs[1], spikes, drives=None)

    axs[1].set_xlim(-1, RUN_LENGTH_NO_DRIVE)
    axs[1].set_ylim(-1, ntwk.w.shape[0])
    axs[1].set_xlabel('time step')
    axs[1].set_ylabel('ensemble')

    # drive the network from two different initial conditions several times
    ntwk = deepcopy(ntwk_base)
    ntwk.store_voltages = True

    # construct drives
    drives = np.zeros((2 * TRIAL_LENGTH * N_REPEATS, ntwk.w.shape[0]), dtype=float)
    node_0_drive_times = np.arange(0, TRIAL_LENGTH * N_REPEATS, TRIAL_LENGTH)
    node_1_drive_times = TRIAL_LENGTH * N_REPEATS + node_0_drive_times
    drives[node_0_drive_times, ntwk.node_0] = DRIVE_AMPLITUDE
    drives[node_1_drive_times, ntwk.node_1] = DRIVE_AMPLITUDE

    for drive in drives:
        ntwk.step(drive)

    spikes = np.array(ntwk.rs_history)

    axs[2].set_title('Variability in spontaneous activity from forced initial condition')
    fancy_raster.by_row_circles(axs[2], spikes, drives)

    axs[2].set_xlim(-1, len(drives))
    axs[2].set_ylim(-1, 20)
    axs[2].set_xlabel('time step')
    axs[2].set_ylabel('ensemble')


def weakly_driven(config):
    """
    Run simulation of activity influenced by weak drive.
    """

    SEED = config['SEED']

    LOAD_FILE_NAME = config['LOAD_FILE_NAME']

    WEAK_DRIVE_AMPLITUDE = config['WEAK_DRIVE_AMPLITUDE']
    STRONG_DRIVE_AMPLITUDE = config['STRONG_DRIVE_AMPLITUDE']

    PATH_MISMATCH_TIME = config['PATH_MISMATCH_TIME']

    TRIAL_LENGTH = config['TRIAL_LENGTH']
    N_REPEATS = config['N_REPEATS']

    FIG_SIZE = config['FIG_SIZE']

    np.random.seed(SEED)

    # load network
    ntwk_base = np.load(LOAD_FILE_NAME)[0]
    n_nodes = ntwk_base.w.shape[0]

    # open figure
    fig, axs = plt.subplots(4, 1, figsize=FIG_SIZE, tight_layout=True)
    
    # pick two paths from same tree
    path_00 = ntwk_base.node_0_path_tree[0]
    path_01 = ntwk_base.node_0_path_tree[1]

    # construct weak drives matching two paths from node 0's path tree
    drives = np.zeros((2 * TRIAL_LENGTH * N_REPEATS, n_nodes), dtype=float)

    for repeat in range(N_REPEATS):

        # calculate offsets for drives corresponding to each node
        t_node_0 = repeat * TRIAL_LENGTH
        t_node_1 = repeat * TRIAL_LENGTH + N_REPEATS * TRIAL_LENGTH

        # set drives at specified timepoint and node to 1
        drives[t_node_0, path_00[0]] = STRONG_DRIVE_AMPLITUDE
        drives[t_node_1, path_01[0]] = STRONG_DRIVE_AMPLITUDE
        for t_ctr, node in enumerate(path_00[1:]):
            drives[t_node_0 + t_ctr + 1, node] = WEAK_DRIVE_AMPLITUDE
        for t_ctr, node in enumerate(path_01[1:]):
            drives[t_node_1 + t_ctr + 1, node] = WEAK_DRIVE_AMPLITUDE

    # present weak drive path-matched to network
    ntwk = deepcopy(ntwk_base)
    ntwk.store_voltages = True

    for drive in drives:
        ntwk.step(drive)

    spikes = np.array(ntwk.rs_history)

    # show network response
    ax = axs[0]
    ax.set_title('Response to weak drive matching anatomical paths')
    fancy_raster.by_row_circles(ax, spikes, drives)

    ax.set_xlim(-1, len(drives))
    ax.set_ylim(-1, 20)
    ax.set_xlabel('time step')
    ax.set_ylabel('active ensemble')

    # construct weak drive matching one path all but time 2, at which it matches path from other tree
    nearly_matching_path = []
    orthogonal_path = ntwk_base.node_1_path_tree[0]
    nearly_matching_path[:] = path_00[:]
    nearly_matching_path[PATH_MISMATCH_TIME] = orthogonal_path[PATH_MISMATCH_TIME]

    drives = np.zeros((TRIAL_LENGTH * N_REPEATS, n_nodes), dtype=float)

    for repeat in range(N_REPEATS):
        t_start = repeat * TRIAL_LENGTH
        drives[t_start, nearly_matching_path[0]] = STRONG_DRIVE_AMPLITUDE
        for t_ctr, node in enumerate(nearly_matching_path[1:]):
            drives[t_start + t_ctr + 1, node] = WEAK_DRIVE_AMPLITUDE

    # present weak drive path-nearly-matched to network
    ntwk = deepcopy(ntwk_base)
    ntwk.store_voltages = True

    for drive in drives:
        ntwk.step(drive)

    spikes = np.array(ntwk.rs_history)

    # show network response
    ax = axs[1]
    ax.set_title('Response to weak drive nearly matching anatomical path')
    fancy_raster.by_row_circles(ax, spikes, drives)

    ax.set_xlim(-1, len(drives))
    ax.set_ylim(-1, 20)
    ax.set_xlabel('time step')
    ax.set_ylabel('active ensemble')

    # construct weak drive matching one path before time 2, and path from other tree after time 2
    half_matching_path = []
    half_matching_path[:] = path_00[:]
    half_matching_path[PATH_MISMATCH_TIME:] = orthogonal_path[PATH_MISMATCH_TIME:]

    drives = np.zeros((TRIAL_LENGTH * N_REPEATS, n_nodes), dtype=float)

    for repeat in range(N_REPEATS):
        t_start = repeat * TRIAL_LENGTH
        drives[t_start, half_matching_path[0]] = STRONG_DRIVE_AMPLITUDE
        for t_ctr, node in enumerate(half_matching_path[1:]):
            drives[t_start + t_ctr + 1, node] = WEAK_DRIVE_AMPLITUDE

    # present weak drive path-half-matched to network
    ntwk = deepcopy(ntwk_base)
    ntwk.store_voltages = True

    for drive in drives:
        ntwk.step(drive)

    spikes = np.array(ntwk.rs_history)

    # show network response
    ax = axs[2]
    ax.set_title('Response to weak drive half matching anatomical path')
    fancy_raster.by_row_circles(ax, spikes, drives)

    ax.set_xlim(-1, len(drives))
    ax.set_ylim(-1, 20)
    ax.set_xlabel('time step')
    ax.set_ylabel('active ensemble')

    # construct strong drive identical to previous weak drive except in strength
    drives[drives > 0] = STRONG_DRIVE_AMPLITUDE

    # present strong drive half-matched to network
    ntwk = deepcopy(ntwk_base)
    ntwk.store_voltages = True

    for drive in drives:
        ntwk.step(drive)

    spikes = np.array(ntwk.rs_history)

    # show network response
    ax = axs[3]
    ax.set_title('Response to strong drive half matching anatomical path')
    fancy_raster.by_row_circles(ax, spikes, drives)

    ax.set_xlim(-1, len(drives))
    ax.set_ylim(-1, 20)
    ax.set_xlabel('time step')
    ax.set_ylabel('active ensemble')