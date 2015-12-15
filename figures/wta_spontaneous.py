from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append('/Users/rkp/Dropbox/Repositories')
sys.path.append('/Users/rkp/Dropbox/Repositories/be_more_spontaneous')
from figure_magic import axis_tools
import network


def main(config):
    # get params from config
    SEED = config['SEED']
    DURATION = config['DURATION']
    
    TAU = config['TAU']
    V_REST = config['V_REST']
    THRESHOLD = config['THRESHOLD']
    STEEPNESS = config['STEEPNESS']
    NOISE_LEVEL = config['NOISE_LEVEL']
    
    W_FS = config['W_FS']
    W_FF = config['W_FF']
    W_FI = config['W_FI']
    
    SWITCH_DRIVE = config['SWITCH_DRIVE']
    
    FONT_SIZE = config['FONT_SIZE']
    
    # setup network parameters; order: (switch, fast, fast_inh)
    w_single = np.array([
        [0, 0, 0],  # to switch
        [W_FS, W_FF, W_FI],  # to fast
        [0, 10, 0],  # to fast_inh
    ])
    nodes_single = 3 * [{'tau': TAU, 'v_rest': V_REST, 'threshold': THRESHOLD, 'steepness': STEEPNESS}]
    
    # build network
    ntwk = network.RateBasedModel(nodes_single, w_single)
    ntwk.noise_level = NOISE_LEVEL
    ntwk.store_voltages = True
    
    # set network drive
    drive = np.array([SWITCH_DRIVE, 0, 0])
    
    # run network
    np.random.seed(SEED)
    ntwk.vs = np.array([V_REST, V_REST, V_REST])
    for t in range(DURATION):
        ntwk.step(drive)
        
    # make figure
    fig, axs = plt.subplots(2, 1, figsize=(15, 5), sharex=True, tight_layout=True)
    axs[0].plot(ntwk.vs_history, lw=2)
    axs[1].plot(ntwk.rs_history, lw=2)
    
    axs[1].set_ylim(-.1, 1.1)
    axs[1].set_yticks(np.linspace(0, 1, 5, endpoint=True))
    axs[0].set_ylabel('Voltage')
    axs[1].set_xlabel('t')
    axs[1].set_ylabel('Firing rate')
    
    for ax in axs:
        axis_tools.set_fontsize(ax, FONT_SIZE)