from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('/Users/rkp/Dropbox/Repositories')
sys.path.append('/Users/rkp/Dropbox/Repositories/be_more_spontaneous')
from figure_magic import axis_tools
import network


TAU = 10
V_RESTING = 0
V_TH = 4
G = 2.5
V_RESTS = [0, 0, 0, -10, -10]
NOISE_LEVEL = 1
SRW = 5.5  # slow recurrent weight
FSW = 3  # fast to slow weight
CGW = 10  # conduit gating weight
CWI = 6  # conduit weight in
CWO = 9  # conduit weight out


FACE_COLOR = 'w'
FIG_SIZE = (14, 10)
FONT_SIZE = 20

SEED = 1


def main():
    # make network
    nodes = [{'tau': TAU, 'v_rest': v_rest, 'threshold': V_TH, 'steepness': G} for v_rest in V_RESTS]
    weights = np.array([
            [0., 0, 0, 0, CWO],
            [0, 0, 0, CWO, 0],
            [FSW, FSW, SRW, 0, 0],
            [CWI, 0, CGW, 0, 0],
            [0, CWI, CGW, 0, 0],
        ])
    ntwk = network.RateBasedModel(nodes, weights)
    ntwk.store_voltages = True
    ntwk.noise_level = NOISE_LEVEL
    
    # set up network drive
    drive_1 = 10*[0] + 20*[10] + 100*[0] + 20*[10] + 80*[0]
    drive_2 = 25*[0] + 20*[10] + 185*[0]
    drive_12s = len(drive_1) * [0]
    drive_12 = len(drive_1) * [0]
    drive_21 = len(drive_1) * [0]
    drives = np.transpose([drive_1, drive_2, drive_12s, drive_12, drive_21])
    t = np.arange(len(drives))
    
    # run network
    np.random.seed(SEED)
    ntwk.vs = np.array(V_RESTS)
    for drive in drives:
        ntwk.step(drive)
    
    # make figure
    fig, axs = plt.subplots(3, 1, facecolor=FACE_COLOR, figsize=FIG_SIZE, sharex=True)
    axs[0].plot(t, ntwk.vs_history, lw=2)
    axs[1].plot(t, ntwk.rs_history, lw=2)
    axs[2].plot(t, drives, lw=2)
    
    axs[0].set_ylabel('voltage')
    axs[1].set_ylabel('firing rate')
    axs[2].set_ylabel('drive')
    axs[2].set_xlabel('t')
    
    axs[0].set_title('Two-component memory network')
    
    for ax in axs:
        axis_tools.set_fontsize(ax, FONT_SIZE)


if __name__ == '__main__':
    main()