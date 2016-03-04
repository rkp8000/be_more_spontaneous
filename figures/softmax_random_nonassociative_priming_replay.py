"""
Demonstrate some of the basic properties of a random network with dynamics
governed by the softmax rule.
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


def basic_replay(CONFIG):
    """
    Run a simulation demonstrating the basic capability of a network with nonassociative priming to
    demonstrated replay, both triggered and spontaneous.
    """

    SEED = CONFIG['SEED']

    LOAD_FILE_NAME = CONFIG['LOAD_FILE_NAME']

    GAIN_HIGH = CONFIG['GAIN_HIGH']
    GAIN_LOW = CONFIG['GAIN_LOW']

    LINGERING_INPUT_VALUE = CONFIG['LINGERING_INPUT_VALUE']
    LINGERING_INPUT_TIMESCALE = CONFIG['LINGERING_INPUT_TIMESCALE']

    STRONG_DRIVE_AMPLITUDE = CONFIG['STRONG_DRIVE_AMPLITUDE']
    WEAK_DRIVE_AMPLITUDE = CONFIG['WEAK_DRIVE_AMPLITUDE']

    TRIAL_LENGTH_TRIGGERED_REPLAY = CONFIG['TRIAL_LENGTH_TRIGGERED_REPLAY']
    RUN_LENGTH = CONFIG['RUN_LENGTH']
    FIG_SIZE = CONFIG['FIG_SIZE']

    np.random.seed(SEED)

    ntwk_base = np.load(LOAD_FILE_NAME)[0]
    ntwk_base.lingering_input_value = LINGERING_INPUT_VALUE
    ntwk_base.lingering_input_timescale = LINGERING_INPUT_TIMESCALE

    fig = plt.figure(figsize=FIG_SIZE, tight_layout=True)
    axs = []
    axs.append(fig.add_subplot(6, 2, 1))
    axs.append(fig.add_subplot(6, 2, 2))
    axs.append(fig.add_subplot(6, 2, 3))
    axs.append(fig.add_subplot(6, 2, 4))
    axs.append(fig.add_subplot(6, 1, 3))
    axs.append(fig.add_subplot(6, 1, 4))
    axs.append(fig.add_subplot(6, 1, 5))
    axs.append(fig.add_subplot(6, 1, 6))

    # play sequences aligned to the network's intrinsic path structure
    path_00 = ntwk_base.node_0_path_tree[0]
    path_10 = ntwk_base.node_1_path_tree[0]

    # drive network for first trial: path_00
    drives = np.zeros((TRIAL_LENGTH_TRIGGERED_REPLAY, ntwk_base.w.shape[1]), dtype=float)
    drives[0, path_00[0]] = STRONG_DRIVE_AMPLITUDE
    for t_ctr, node in enumerate(path_00[1:]):
        drives[t_ctr + 1, node] = WEAK_DRIVE_AMPLITUDE
    drives[len(path_00), path_00[0]] = STRONG_DRIVE_AMPLITUDE

    ntwk = deepcopy(ntwk_base)
    ntwk.gain = GAIN_HIGH
    ntwk.store_voltages = True

    for drive in drives:
        ntwk.step(drive)

    spikes = np.array(ntwk.rs_history)

    fancy_raster.by_row_circles(axs[0], spikes, drives)

    axs[0].set_xlim(-1, len(drives))
    axs[0].set_ylim(-1, 20)
    axs[0].set_xlabel('time step')
    axs[0].set_ylabel('active ensemble')
    axs[0].set_title('Aligning external drive with \n strongly connected paths')

    # drive network for first trial: path_10
    drives = np.zeros((TRIAL_LENGTH_TRIGGERED_REPLAY, ntwk_base.w.shape[1]), dtype=float)
    drives[0, path_10[0]] = STRONG_DRIVE_AMPLITUDE
    for t_ctr, node in enumerate(path_10[1:]):
        drives[t_ctr + 1, node] = WEAK_DRIVE_AMPLITUDE
    drives[len(path_10), path_10[0]] = STRONG_DRIVE_AMPLITUDE

    ntwk = deepcopy(ntwk_base)
    ntwk.gain = GAIN_HIGH
    ntwk.store_voltages = True

    for drive in drives:
        ntwk.step(drive)

    spikes = np.array(ntwk.rs_history)

    fancy_raster.by_row_circles(axs[1], spikes, drives)

    axs[1].set_xlim(-1, len(drives))
    axs[1].set_ylim(-1, 20)
    axs[1].set_xlabel('time step')
    axs[1].set_ylabel('active ensemble')
    axs[1].set_title('Aligning external drive with \n strongly connected paths')

    # drive network for third trial: all path_00 except for element 2
    path = list(path_00[:])
    path[2] = path_10[2]
    drives = np.zeros((TRIAL_LENGTH_TRIGGERED_REPLAY, ntwk_base.w.shape[1]), dtype=float)
    drives[0, path[0]] = STRONG_DRIVE_AMPLITUDE
    for t_ctr, node in enumerate(path[1:]):
        drives[t_ctr + 1, node] = WEAK_DRIVE_AMPLITUDE
    drives[len(path), path[0]] = STRONG_DRIVE_AMPLITUDE

    ntwk = deepcopy(ntwk_base)
    ntwk.gain = GAIN_HIGH
    ntwk.store_voltages = True

    for drive in drives:
        ntwk.step(drive)

    spikes = np.array(ntwk.rs_history)

    fancy_raster.by_row_circles(axs[2], spikes, drives)

    axs[2].set_xlim(-1, len(drives))
    axs[2].set_ylim(-1, 20)
    axs[2].set_xlabel('time step')
    axs[2].set_ylabel('active ensemble')
    axs[2].set_title('Aligning external drive with \n nonexisting path')

    # drive network for fourth trial: all path_10 except for element 2
    path = list(path_10[:])
    path[2] = path_00[2]
    drives = np.zeros((TRIAL_LENGTH_TRIGGERED_REPLAY, ntwk_base.w.shape[1]), dtype=float)
    drives[0, path[0]] = STRONG_DRIVE_AMPLITUDE
    for t_ctr, node in enumerate(path[1:]):
        drives[t_ctr + 1, node] = WEAK_DRIVE_AMPLITUDE
    drives[len(path), path[0]] = STRONG_DRIVE_AMPLITUDE

    ntwk = deepcopy(ntwk_base)
    ntwk.gain = GAIN_HIGH
    ntwk.store_voltages = True

    for drive in drives:
        ntwk.step(drive)

    spikes = np.array(ntwk.rs_history)

    fancy_raster.by_row_circles(axs[3], spikes, drives)

    axs[3].set_xlim(-1, len(drives))
    axs[3].set_ylim(-1, 20)
    axs[3].set_xlabel('time step')
    axs[3].set_ylabel('active ensemble')
    axs[3].set_title('Aligning external drive with \n nonexisting path')

    # play sequence and then let network run spontaneously for a while
    drives = np.zeros((RUN_LENGTH, ntwk_base.w.shape[1]), dtype=float)
    for t_ctr, node in enumerate(path_00):
        drives[t_ctr, node] = STRONG_DRIVE_AMPLITUDE

    ntwk = deepcopy(ntwk_base)
    ntwk.gain = GAIN_HIGH
    ntwk.store_voltages = True

    for drive in drives:
        ntwk.step(drive)

    spikes = np.array(ntwk.rs_history)

    fancy_raster.by_row_circles(axs[4], spikes, drives)

    axs[4].set_xlim(-1, len(drives))
    axs[4].set_ylim(-1, ntwk_base.w.shape[1])
    axs[4].set_xlabel('time step')
    axs[4].set_ylabel('active ensemble')
    axs[4].set_title('Letting network run freely after driving strongly connected path (high gain)')

    # play sequence and then let network run spontaneously for a while, now with lower gain
    drives = np.zeros((RUN_LENGTH, ntwk_base.w.shape[1]), dtype=float)
    for t_ctr, node in enumerate(path_00):
        drives[t_ctr, node] = STRONG_DRIVE_AMPLITUDE

    ntwk = deepcopy(ntwk_base)
    ntwk.gain = GAIN_LOW
    ntwk.store_voltages = True

    for drive in drives:
        ntwk.step(drive)

    spikes = np.array(ntwk.rs_history)

    fancy_raster.by_row_circles(axs[5], spikes, drives)

    axs[5].set_xlim(-1, len(drives))
    axs[5].set_ylim(-1, ntwk_base.w.shape[1])
    axs[5].set_xlabel('time step')
    axs[5].set_ylabel('active ensemble')
    axs[5].set_title('Letting network run freely after driving strongly connected path (low gain)')

    # let network run spontaneously for a while with no initial drive
    ntwk = deepcopy(ntwk_base)
    ntwk.gain = GAIN_HIGH
    ntwk.store_voltages = True

    for _ in range(RUN_LENGTH):
        ntwk.step()

    spikes = np.array(ntwk.rs_history)

    fancy_raster.by_row_circles(axs[6], spikes, drives=None)

    axs[6].set_xlim(-1, len(drives))
    axs[6].set_ylim(-1, ntwk_base.w.shape[1])
    axs[6].set_xlabel('time step')
    axs[6].set_ylabel('active ensemble')
    axs[6].set_title('Letting network run freely with no drive (high gain)')

    # let network run spontaneously for a while with no initial drive, now with lower gain
    ntwk = deepcopy(ntwk_base)
    ntwk.gain = GAIN_LOW
    ntwk.store_voltages = True

    for _ in range(RUN_LENGTH):
        ntwk.step()

    spikes = np.array(ntwk.rs_history)

    fancy_raster.by_row_circles(axs[7], spikes, drives=None)

    axs[7].set_xlim(-1, len(drives))
    axs[7].set_ylim(-1, ntwk_base.w.shape[1])
    axs[7].set_xlabel('time step')
    axs[7].set_ylabel('active ensemble')
    axs[7].set_title('Letting network run freely with no drive (low gain)')


def novel_pattern_replay(CONFIG):
    """
    Show how a network that has weak connections in addition to its strong connections can learn to replay novel patterns that would not normally arise spontaneously.
    """

    SEED = CONFIG['SEED']

    LOAD_FILE_NAME = CONFIG['LOAD_FILE_NAME']

    W_WEAK = CONFIG['W_WEAK']

    GAIN = CONFIG['GAIN']
    REFRACTORY_STRENGTH = CONFIG['REFRACTORY_STRENGTH']

    LINGERING_INPUT_VALUE = CONFIG['LINGERING_INPUT_VALUE']
    LINGERING_INPUT_TIMESCALE = CONFIG['LINGERING_INPUT_TIMESCALE']

    STRONG_DRIVE_AMPLITUDE = CONFIG['STRONG_DRIVE_AMPLITUDE']
    WEAK_DRIVE_AMPLITUDE = CONFIG['WEAK_DRIVE_AMPLITUDE']

    TRIAL_LENGTH_TRIGGERED_REPLAY = CONFIG['TRIAL_LENGTH_TRIGGERED_REPLAY']
    RUN_LENGTH = CONFIG['RUN_LENGTH']

    FIG_SIZE_0 = CONFIG['FIG_SIZE_0']
    FIG_SIZE_1 = CONFIG['FIG_SIZE_1']

    np.random.seed(SEED)

    fig = plt.figure(figsize=FIG_SIZE_0, tight_layout=True)
    axs = []
    for row_ctr in range(3):
        axs.append([fig.add_subplot(4, 3, 3*row_ctr + col_ctr) for col_ctr in range(1, 4)])
    axs = list(axs)
    axs.append(fig.add_subplot(4, 1, 4))

    # load old network
    ntwk_old = np.load(LOAD_FILE_NAME)[0]

    # demonstrate how one cannot use intrinsic plasticity to learn sequence that is made of disjoint paths
    path = list(ntwk_old.node_0_path_tree[0][:])
    path[2:] = ntwk_old.node_1_path_tree[0][2:]
    drives = np.zeros((TRIAL_LENGTH_TRIGGERED_REPLAY, ntwk_old.w.shape[0]), dtype=float)
    for ctr, node in enumerate(path):
        drives[ctr, node] = STRONG_DRIVE_AMPLITUDE
    drives[len(path), path[0]] = STRONG_DRIVE_AMPLITUDE

    for ctr, ax in enumerate(axs[0]):
        ntwk = deepcopy(ntwk_old)
        ntwk.store_voltages = True

        for drive in drives:
            ntwk.step(drive)

        spikes = np.array(ntwk.rs_history)

        fancy_raster.by_row_circles(ax, spikes, drives)

        ax.set_xlim(-1, len(drives))
        ax.set_ylim(-1, 20)
        ax.set_xlabel('time step')
        ax.set_ylabel('active ensemble')
        ax.set_title('Strongly driving nonexisting path (trial {})'.format(ctr + 1))

    w = ntwk_old.w.copy()

    # add weak connection to element 2 of node_1 path tree from element 1 of node_0 path tree
    w[ntwk_old.node_1_path_tree[0][2], ntwk_old.node_0_path_tree[0][1]] = W_WEAK

    # make new base network
    ntwk_base = network.RecurrentSoftMaxLingeringModel(
        w, GAIN, REFRACTORY_STRENGTH, LINGERING_INPUT_VALUE, LINGERING_INPUT_TIMESCALE
    )
    ntwk_base.node_0 = ntwk_old.node_0
    ntwk_base.node_0 = ntwk_old.node_1
    ntwk_base.node_0_path_tree = ntwk_old.node_0_path_tree
    ntwk_base.node_1_path_tree = ntwk_old.node_1_path_tree

    # demonstrate how weak connections allow linking of paths into short term memory
    path = list(ntwk_base.node_0_path_tree[0][:])
    path[2:] = ntwk_base.node_1_path_tree[0][2:]
    drives = np.zeros((TRIAL_LENGTH_TRIGGERED_REPLAY, ntwk_base.w.shape[0]), dtype=float)
    for ctr, node in enumerate(path):
        drives[ctr, node] = STRONG_DRIVE_AMPLITUDE
    drives[len(path), path[0]] = STRONG_DRIVE_AMPLITUDE

    for ctr, ax in enumerate(axs[1]):
        ntwk = deepcopy(ntwk_base)
        ntwk.store_voltages = True

        for drive in drives:
            ntwk.step(drive)

        spikes = np.array(ntwk.rs_history)

        fancy_raster.by_row_circles(ax, spikes, drives)

        ax.set_xlim(-1, len(drives))
        ax.set_ylim(-1, 20)
        ax.set_xlabel('time step')
        ax.set_ylabel('active ensemble')
        ax.set_title('Activity from forced initial condition after \n driving path with weak connection (trial {})'.format(ctr + 1))

    # demonstrate how weak connections do not substantially affect path probabilities
    path = list(ntwk_base.node_0_path_tree[0][:])
    path[2:] = ntwk_base.node_1_path_tree[0][2:]
    drives = np.zeros((TRIAL_LENGTH_TRIGGERED_REPLAY, ntwk_old.w.shape[0]), dtype=float)
    drives[0, path[0]] = STRONG_DRIVE_AMPLITUDE

    for ctr, ax in enumerate(axs[2]):
        ntwk = deepcopy(ntwk_base)
        ntwk.store_voltages = True

        for drive in drives:
            ntwk.step(drive)

        spikes = np.array(ntwk.rs_history)

        fancy_raster.by_row_circles(ax, spikes, drives)

        ax.set_xlim(-1, len(drives))
        ax.set_ylim(-1, 20)
        ax.set_xlabel('time step')
        ax.set_ylabel('active ensemble')
        ax.set_title('Free activity after only forcing \n initial condition (trial {})'.format(ctr + 1))

    path = list(ntwk_base.node_0_path_tree[0][:])
    path[2:] = ntwk_base.node_1_path_tree[0][2:]
    drives = np.zeros((RUN_LENGTH, ntwk_base.w.shape[0]), dtype=float)
    for ctr, node in enumerate(path):
        drives[ctr, node] = STRONG_DRIVE_AMPLITUDE

    ntwk = deepcopy(ntwk_base)
    ntwk.store_voltages = True
    for drive in drives:
        ntwk.step(drive)

    spikes = np.array(ntwk.rs_history)

    fancy_raster.by_row_circles(axs[3], spikes, drives)

    axs[3].set_xlim(-1, len(drives))
    axs[3].set_ylim(-1, 40)
    axs[3].set_xlabel('time step')
    axs[3].set_ylabel('active ensemble')
    axs[3].set_title('Free activity after driving path with weak connection')

    # now demonstrate how pattern-matching computation changes with respect to short-term memory
    fig, axs = plt.subplots(1, 2, figsize=FIG_SIZE_1, tight_layout=True)

    path = list(ntwk_base.node_0_path_tree[1][:])
    path[0] = 22
    path[2] = 17
    path[3] = ntwk_base.node_1_path_tree[0][3]

    drives_new = np.zeros((len(path), ntwk_base.w.shape[0]), dtype=float)
    drives_new[0, path[0]] = STRONG_DRIVE_AMPLITUDE
    for ctr, node in enumerate(path[1:]):
        drives_new[ctr + 1, node] = WEAK_DRIVE_AMPLITUDE

    # drive a network with just the new drive to see how it completes the pattern
    ntwk = deepcopy(ntwk_base)
    ntwk.store_voltages = True
    for drive in drives_new:
        ntwk.step(drive)

    spikes = np.array(ntwk.rs_history)

    fancy_raster.by_row_circles(axs[0], spikes, drives_new)

    axs[0].set_xlim(-1, 8)
    axs[0].set_ylim(-1, ntwk_base.w.shape[0])
    axs[0].set_xlabel('time step')
    axs[0].set_ylabel('active ensemble')
    axs[0].set_title('Weakly driving nonexistent path')

    drives = np.concatenate([drives[:4, :], drives_new])

    ntwk = deepcopy(ntwk_base)
    ntwk.store_voltages = True
    for drive in drives:
        ntwk.step(drive)

    spikes = np.array(ntwk.rs_history)

    fancy_raster.by_row_circles(axs[1], spikes, drives)

    axs[1].set_xlim(-1, 8)
    axs[1].set_ylim(-1, ntwk_base.w.shape[0])
    axs[1].set_xlabel('time step')
    axs[1].set_ylabel('active ensemble')
    axs[1].set_title('Weakly driving nonexistent path after \n strongly driving path with weak connection')