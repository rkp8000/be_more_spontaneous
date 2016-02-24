"""
Demonstrate some of the basic properties of a random network with dynamics
governed by the softmax rule.
"""
from __future__ import division, print_function
from copy import deepcopy
from more_itertools import unique_everseen
import matplotlib.pyplot as plt
import numpy as np


import fancy_raster
import metrics
import network


def spontaneous(config):
    """
    Run simulation of spontaneous activity.
    """

    SEED = config['SEED']

    N_NODES = config['N_NODES']
    P_CXN = config['P_CXN']

    W_STRONG = config['W_STRONG']

    GAIN = config['GAIN']

    PATH_LENGTH = config['PATH_LENGTH']

    RUN_LENGTH_NO_DRIVE = config['RUN_LENGTH_NO_DRIVE']
    TRIAL_LENGTH = config['TRIAL_LENGTH']
    N_REPEATS = config['N_REPEATS']

    FIG_SIZE_NO_DRIVE = config['FIG_SIZE_NO_DRIVE']
    FIG_SIZE_WITH_DRIVE = config['FIG_SIZE_WITH_DRIVE']

    LOAD_FILE_NAME = config['LOAD_FILE_NAME']
    SAVE_FILE_NAME = config['SAVE_FILE_NAME']

    np.random.seed(SEED)

    if LOAD_FILE_NAME:
        ntwk_base = np.load(LOAD_FILE_NAME)[0]

    else:
        # loop over attempts to make network with desired properties
        while True:
            # make a random weight matrix
            weights = W_STRONG * (np.random.rand(N_NODES, N_NODES) < P_CXN).astype(float)
            np.fill_diagonal(weights, 0)

            # make sure it has at least two nodes whose path trees don't overlap
            paths, _ = metrics.most_probable_paths(weights, GAIN, PATH_LENGTH, None)
            prioritized_start_nodes = unique_everseen([path[0] for path in paths])
            found_start_nodes = metrics.first_node_pair_non_overlapping_path_tree(
                prioritized_start_nodes, weights, PATH_LENGTH, allow_path_loops=False,
            )

            # make sure that at least one path tree contains two distinct paths
            if found_start_nodes:

                node_0, node_1 = found_start_nodes

                node_0_path_tree = metrics.paths_of_length(weights, PATH_LENGTH, node_0)
                node_1_path_tree = metrics.paths_of_length(weights, PATH_LENGTH, node_1)

                if len(node_0_path_tree) > 1:
                    break

                if len(node_1_path_tree) > 1:
                    # relabel things so node
                    node_0, node_1 = node_1, node_0
                    break

        # reorder columns and rows of weight matrix according to spotlighted path trees
        weights, ordering = metrics.reorder_by_paths(weights, paths)
        node_0 = ordering.index(node_0)
        node_1 = ordering.index(node_1)

        # make a network with this weight matrix
        ntwk_base = network.RecurrentSoftMaxModel(weights, GAIN)

        # bind two root nodes and path trees to network
        ntwk_base.node_0 = node_0
        ntwk_base.node_1 = node_1
        ntwk_base.node_0_path_tree = metrics.paths_of_length(weights, PATH_LENGTH, node_0)
        ntwk_base.node_1_path_tree = metrics.paths_of_length(weights, PATH_LENGTH, node_1)

        # save the network
        np.save(SAVE_FILE_NAME, [ntwk_base], allow_pickle=True)

    # let the network run spontaneously
    ntwk = deepcopy(ntwk_base)
    ntwk.store_voltages = True

    for _ in range(RUN_LENGTH_NO_DRIVE):
        ntwk.step()

    spikes = np.array(ntwk.rs_history)

    fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE_NO_DRIVE, tight_layout=True)
    fancy_raster.by_row(ax, spikes, drives=None)

    # drive the network from two different initial conditions several times
    ntwk = deepcopy(ntwk_base)
    ntwk.store_voltages = True

    # construct drives
    drives = np.zeros((2 * TRIAL_LENGTH * N_REPEATS, N_NODES), dtype=float)
    node_0_drive_times = np.arange(TRIAL_LENGTH * N_REPEATS, TRIAL_LENGTH)
    node_1_drive_times = TRIAL_LENGTH * N_REPEATS + node_0_drive_times
    drives[node_0_drive_times, ntwk.node_0] = 1
    drives[node_1_drive_times, ntwk.node_1] = 1

    for drive in drives:
        ntwk.step(10 * drive)

    spikes = np.array(ntwk.rs_history)

    fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE_WITH_DRIVE, tight_layout=True)
    fancy_raster.by_row(ax, spikes, drives)

    # show how changes in gain affect randomness???


def weakly_driven(config):
    """
    Run simulation of activity influenced by weak drive.
    """

    SEED = config['SEED']

    LOAD_FILE_NAME = config['LOAD_FILE_NAME']

    PATH_MISMATCH_TIME = config['PATH_MISMATCH_TIME']

    N_REPEATS = config['N_REPEATS']

    # load network

    # construct weak drives matching two paths in path tree

    # construct weak drive matching one path all but time 2, at which it matches path from other tree

    # construct weak drive matching one path before time 2, and path from other tree after time 2

    # construct strong drive identical to previous weak drive except in strength

    # present path-matched weak drives to network

    # present path-nearly-matched weak drive to network

    # present path-half-matched weak drive to network

    # present path-half-matched strong drive to network