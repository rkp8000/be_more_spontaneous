"""
Demonstrate some of the basic properties of a random network with dynamics
governed by the softmax rule.
"""
from __future__ import division, print_function
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append('/Users/rkp/Dropbox/Repositories/be_more_spontaneous')


import fancy_raster


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
    for t_ctr, node in enumerate(path_00):
        drives[t_ctr, node] = STRONG_DRIVE_AMPLITUDE
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

    # drive network for first trial: path_10
    drives = np.zeros((TRIAL_LENGTH_TRIGGERED_REPLAY, ntwk_base.w.shape[1]), dtype=float)
    for t_ctr, node in enumerate(path_10):
        drives[t_ctr, node] = STRONG_DRIVE_AMPLITUDE
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
    axs[4].set_title('high gain')

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
    axs[5].set_title('low gain')

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
    axs[6].set_title('high gain')

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
    axs[7].set_title('low gain')