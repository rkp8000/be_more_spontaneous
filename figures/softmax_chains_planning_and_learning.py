from __future__ import division, print_function
from copy import deepcopy as copy
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import warnings
warnings.filterwarnings('ignore')

import axis_tools
import network
import network_param_gen


def planning(config):
    """
    Simulation of sequence "planning" via purely spatial inputs.
    """

    # parse config parameters
    SEED = config['SEED']

    N_CHAINS = config['N_CHAINS']
    CHAIN_LENGTH = config['CHAIN_LENGTH']

    W_STRONG = config['W_STRONG']
    W_WEAK = config['W_WEAK']
    WEAK_CXN_IDXS = config['WEAK_CXN_IDXS']

    GAIN = config['GAIN']
    HDE_INPUT_VALUE = config['HDE_INPUT_VALUE']

    DRIVE_MULTI_BY_COORDINATE = config['DRIVE_MULTI_BY_COORDINATE']

    N_TRIALS = config['N_TRIALS']

    FIG_SIZE = config['FIG_SIZE']
    COLORS = config['COLORS']
    LW = config['LW']
    FONT_SIZE = config['FONT_SIZE']

    shape = (N_CHAINS, CHAIN_LENGTH)

    # make chain weight matrix
    weights = W_STRONG * network_param_gen.chain_weight_matrix(
        n_chains=N_CHAINS, chain_length=CHAIN_LENGTH,
    )

    # add connections between pairs of chains
    for chain_idx in range(N_CHAINS - 1):
        # get id of target node
        targ_id = np.ravel_multi_index((chain_idx + 1, WEAK_CXN_IDXS[0]), shape)
        # get id of source node
        src_id = np.ravel_multi_index((chain_idx, WEAK_CXN_IDXS[1]), shape)
        weights[targ_id, src_id] = W_WEAK

    # make base network
    ntwk_base = network.RecurrentSoftMaxLingeringModel(
        weights=weights, gain=GAIN, lingering_input_value=HDE_INPUT_VALUE, shape=(N_CHAINS, CHAIN_LENGTH),
    )
    ntwk_base.store_voltages = True

    # run networks and store output firing rates
    drives_plottables = []
    rs_plottables = []
    np.random.seed(SEED)

    for t_ctr in range(N_TRIALS):

        ntwk = copy(ntwk_base)

        # initialize the network to start at zero voltage
        ntwk.vs = np.zeros((N_CHAINS * CHAIN_LENGTH,), dtype=float)

        drives_matrix = []

        # run for drive provided
        for drive_multi in DRIVE_MULTI_BY_COORDINATE[t_ctr]:

            drive = np.zeros((N_CHAINS, CHAIN_LENGTH), dtype=float)

            for node_coord, amplitude in drive_multi:
                drive[node_coord] = amplitude

            drives_matrix.append(drive)
            ntwk.step(drive.flatten())

        # do a bit of reshaping on drives to get plottable arrays
        drives_matrix = np.array(drives_matrix)
        drives_plottables.append([drives_matrix[:, :, chain_pos] for chain_pos in range(CHAIN_LENGTH)])

        # same for responses
        rs_matrix = np.array(
            [r.reshape(N_CHAINS, CHAIN_LENGTH) for r in ntwk.rs_history]
        )
        rs_plottables.append([rs_matrix[:, :, chain_pos] for chain_pos in range(CHAIN_LENGTH)])

    # plot activity history for all trials
    fig, axs = plt.subplots(
        CHAIN_LENGTH, N_TRIALS, figsize=FIG_SIZE,
        sharex=True, sharey=True, tight_layout=True
    )

    axs_twin = np.zeros(axs.shape, dtype=object)
    for t_ctr in range(N_TRIALS):

        for ctr, (drives, rs, ax) in enumerate(zip(drives_plottables[t_ctr], rs_plottables[t_ctr], axs[:, t_ctr])):

            if ctr == 0:
                ax.set_title('Trial {}'.format(t_ctr))

            ax.set_color_cycle(COLORS)
            ax.plot(rs, lw=LW)

            ax_twin = ax.twinx()
            ax_twin.set_color_cycle(COLORS)
            ax_twin.plot(drives, lw=LW, ls='--')
            axs_twin[ctr, t_ctr] = ax_twin

    for ax in axs[:, 0]:
        ax.set_ylabel('Rate')

    for ax in axs_twin[:, -1]:
        ax.set_ylabel('Drive')

    for ax in axs[-1, :]:
        ax.set_xlabel('Time')

    for ax in axs_twin.flatten():
        ax.set_ylim(0, 4)
        ax.set_yticks([0, 2, 4])
        axis_tools.set_fontsize(ax, FONT_SIZE)

    for ax in axs.flatten():
        ax.set_xlim(0, len(DRIVE_MULTI_BY_COORDINATE[0]) - 1)
        ax.set_ylim(0, 1)
        ax.set_xticks(range(len(DRIVE_MULTI_BY_COORDINATE[0])))
        axis_tools.set_fontsize(ax, FONT_SIZE)


def learning(config):
    """
    Simulation of learning new sequences via STDP and history-dependent excitability.
    """
    # parse config parameters
    SEED = config['SEED']

    N_CHAINS = config['N_CHAINS']
    CHAIN_LENGTH = config['CHAIN_LENGTH']

    W_STRONG = config['W_STRONG']
    W_WEAK = config['W_WEAK']
    WEAK_CXN_IDXS = config['WEAK_CXN_IDXS']

    GAIN = config['GAIN']
    HDE_INPUT_VALUE = config['HDE_INPUT_VALUE']
    HDE_TIMESCALE = config['HDE_TIMESCALE']

    ALPHA = config['ALPHA']
    W_MAX = config['W_MAX']

    DRIVE_BY_COORDINATE = config['DRIVE_BY_COORDINATE']

    N_TRIALS = config['N_TRIALS']
    WS_TO_TRACK = config['WS_TO_TRACK']

    FIG_SIZE = config['FIG_SIZE']
    COLORS = config['COLORS']
    LW = config['LW']
    FONT_SIZE = config['FONT_SIZE']

    shape = (N_CHAINS, CHAIN_LENGTH)

    # get flattened ids of weights to track
    ws_to_track = [
        (np.ravel_multi_index(w_0, shape), np.ravel_multi_index(w_1, shape))
        for w_0, w_1 in WS_TO_TRACK
    ]

    # make chain weight matrix
    weights = W_STRONG * network_param_gen.chain_weight_matrix(
        n_chains=N_CHAINS, chain_length=CHAIN_LENGTH,
    )

    # add connections between pairs of chains
    for chain_idx in range(N_CHAINS - 1):
        # get id of source node
        targ_id = np.ravel_multi_index((chain_idx + 1, WEAK_CXN_IDXS[0]), shape)
        # get id of target node
        src_id = np.ravel_multi_index((chain_idx, WEAK_CXN_IDXS[1]), shape)
        weights[targ_id, src_id] = W_WEAK

    ntwk_base = network.RecurrentSoftMaxLingeringSTDPModel(
        weights=weights, gain=GAIN, lingering_input_value=HDE_INPUT_VALUE, lingering_timescale=HDE_TIMESCALE,
        w_max=W_MAX, alpha=ALPHA, shape=(N_CHAINS, CHAIN_LENGTH),
    )
    ntwk_base.store_voltages = True

    drives_plottables = []
    rs_plottables = []
    ws_plottables = []
    np.random.seed(SEED)

    for t_ctr in range(N_TRIALS):

        ntwk = copy(ntwk_base)

        ntwk.vs = np.zeros((N_CHAINS * CHAIN_LENGTH,), dtype=float)

        drives_matrix = []
        ws_this_trial = []

        # run for drive provided
        for node_coord, amplitude in DRIVE_BY_COORDINATE[t_ctr]:

            drive = np.zeros((N_CHAINS, CHAIN_LENGTH), dtype=float)
            drive[node_coord] = amplitude
            drives_matrix.append(drive)

            ws_this_trial.append([ntwk.w[idx_0, idx_1] for idx_0, idx_1 in ws_to_track])

            ntwk.step(drive.flatten())

        # do a bit of reshaping on drives to get plottable arrays
        drives_matrix = np.array(drives_matrix)
        drives_plottables.append([drives_matrix[:, :, chain_pos] for chain_pos in range(CHAIN_LENGTH)])
        # save relevant connection weights
        ws_this_trial = np.array(ws_this_trial)
        ws_plottables.append(ws_this_trial)

        # same for responses
        rs_matrix = np.array(
            [r.reshape(N_CHAINS, CHAIN_LENGTH) for r in ntwk.rs_history]
        )
        rs_plottables.append([rs_matrix[:, :, chain_pos] for chain_pos in range(CHAIN_LENGTH)])

    for t_ctr in range(N_TRIALS):

        fig, axs = plt.subplots(CHAIN_LENGTH + 1, 1, figsize=FIG_SIZE, sharex=True, tight_layout=True)
        fig.suptitle('Trial {}'.format(t_ctr), fontsize=FONT_SIZE)

        axs_twin = np.zeros((CHAIN_LENGTH,), dtype=object)

        for ctr, (drives, rs, ax) in enumerate(zip(drives_plottables[t_ctr], rs_plottables[t_ctr], axs[:-1])):

            ax.set_color_cycle(COLORS)
            ax.plot(rs, lw=LW)

            ax_twin = ax.twinx()
            ax_twin.set_color_cycle(COLORS)
            ax_twin.plot(drives, lw=LW, ls='--')
            axs_twin[ctr] = ax_twin

        # plot weights on last axis object
        axs[-1].plot(ws_plottables[t_ctr], lw=LW)

        for ax in axs[:-1]:
            ax.set_ylabel('Rate')

        for ax in axs_twin:
            ax.set_ylabel('Drive')

        axs[-1].set_xlabel('Time')
        axs[-1].set_ylabel('CXN strength')

        for ax in axs_twin:
            ax.set_ylim(0, 4)
            ax.set_yticks([0, 2, 4])
            axis_tools.set_fontsize(ax, FONT_SIZE)

        for ax in axs[:-1].flatten():
            ax.set_xlim(0, len(DRIVE_BY_COORDINATE[0]) - 1)
            ax.set_ylim(0, 1)

        axs[-1].set_ylim(0, W_MAX * 1.1)

        for ax in axs:
            axis_tools.set_fontsize(ax, FONT_SIZE)