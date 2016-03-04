"""
Show how adding a "poor man's" STDP rule can allow nonassociative priming
network to permanently embed new novel sequences into anatomical connectivity.
"""
from __future__ import division, print_function
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.append('/Users/rkp/Dropbox/Repositories/be_more_spontaneous')

import fancy_raster
import network


def novel_pattern_learning(CONFIG):
    """
    Show that a network has an increased probability of embedding a novel pattern
    into its connectivity network if we allow nonassociative priming to act.
    """

    SEED = CONFIG['SEED']

    LOAD_FILE_NAME = CONFIG['LOAD_FILE_NAME']

    W_WEAK = CONFIG['W_WEAK']

    GAIN = CONFIG['GAIN']
    REFRACTORY_STRENGTH = CONFIG['REFRACTORY_STRENGTH']

    LINGERING_INPUT_VALUE = CONFIG['LINGERING_INPUT_VALUE']
    LINGERING_INPUT_TIMESCALE = CONFIG['LINGERING_INPUT_TIMESCALE']

    W_MAX = CONFIG['W_MAX']
    ALPHA = CONFIG['ALPHA']

    STRONG_DRIVE_AMPLITUDE = CONFIG['STRONG_DRIVE_AMPLITUDE']

    RUN_LENGTH = CONFIG['RUN_LENGTH']

    N_REPEATS = CONFIG['N_REPEATS']

    FIG_SIZE_0 = CONFIG['FIG_SIZE_0']

    np.random.seed(SEED)

    # create new network with STDP learning rule
    ntwk_old = np.load(LOAD_FILE_NAME)[0]

    w = ntwk_old.w.copy()

    # add weak connection to element 2 of node_1 path tree from element 1 of node_0 path tree
    w[ntwk_old.node_1_path_tree[0][2], ntwk_old.node_0_path_tree[0][1]] = W_WEAK
    w_to_track = (ntwk_old.node_1_path_tree[0][2], ntwk_old.node_0_path_tree[0][1])
    path_novel = ntwk_old.node_0_path_tree[0][:2] + ntwk_old.node_1_path_tree[0][2:]

    # make new base network
    ntwk_base = network.RecurrentSoftMaxLingeringSTDPModelBasic(
        w, GAIN, REFRACTORY_STRENGTH, LINGERING_INPUT_VALUE, LINGERING_INPUT_TIMESCALE, W_MAX, ALPHA,
    )
    ntwk_base.node_0 = ntwk_old.node_0
    ntwk_base.node_0 = ntwk_old.node_1
    ntwk_base.node_0_path_tree = ntwk_old.node_0_path_tree
    ntwk_base.node_1_path_tree = ntwk_old.node_1_path_tree

    # show long trials of novel sequence drive, spontaneous activity, and test sequence
    fig, axs = plt.subplots(N_REPEATS, 1, figsize=FIG_SIZE_0, sharex=True, tight_layout=True)
    axs_twin = [ax.twinx() for ax in axs]

    drives = np.zeros((RUN_LENGTH, ntwk_base.w.shape[0]), dtype=float)
    for ctr, node in enumerate(path_novel):
        drives[ctr, node] = STRONG_DRIVE_AMPLITUDE

    for ax, ax_twin in zip(axs, axs_twin):

        ntwk = deepcopy(ntwk_base)
        ntwk.store_voltages = True

        ws = []

        for drive in drives:
            ntwk.step(drive)
            ws.append(ntwk.w[w_to_track])

        spikes = np.array(ntwk.rs_history)

        fancy_raster.by_row_circles(ax, spikes, drives)

        ax_twin.plot(ws, color='b', lw=2, alpha=0.7)

        ax.set_ylabel('Active ensemble')
        ax_twin.set_ylabel('W({}, {})'.format(*w_to_track), color='b')

    axs[-1].set_xlabel('time step')

    for ax_twin in axs_twin:
        ax_twin.set_ylim(0, 2)

    axs[0].set_xlim(-5, RUN_LENGTH)
    axs[0].set_title('With nonassociative priming')

    fig, axs = plt.subplots(N_REPEATS, 1, figsize=FIG_SIZE_0, sharex=True, tight_layout=True)
    axs_twin = [ax.twinx() for ax in axs]

    drives = np.zeros((RUN_LENGTH, ntwk_base.w.shape[0]), dtype=float)
    for ctr, node in enumerate(path_novel):
        drives[ctr, node] = STRONG_DRIVE_AMPLITUDE

    for ax, ax_twin in zip(axs, axs_twin):

        ntwk = deepcopy(ntwk_base)
        ntwk.lingering_input_value = 0
        ntwk.store_voltages = True

        ws = []

        for drive in drives:
            ntwk.step(drive)
            ws.append(ntwk.w[w_to_track])

        spikes = np.array(ntwk.rs_history)

        fancy_raster.by_row_circles(ax, spikes, drives)

        ax_twin.plot(ws, color='b', lw=2, alpha=0.7)

        ax.set_ylabel('Active ensemble')
        ax_twin.set_ylabel('W({}, {})'.format(*w_to_track), color='b')

    axs[-1].set_xlabel('time step')

    for ax_twin in axs_twin:
        ax_twin.set_ylim(0, 2)

    axs[0].set_xlim(-5, RUN_LENGTH)
    axs[0].set_title('Without nonassociative priming')