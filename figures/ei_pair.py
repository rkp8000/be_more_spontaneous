from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append('/Users/rkp/Dropbox/Repositories')
sys.path.append('/Users/rkp/Dropbox/Repositories/be_more_spontaneous')
from figure_magic import axis_tools
import network


def main(config):
    SEED = config['SEED']
    
    TAU_E = config['TAU_E']
    TAU_I = config['TAU_I']
    V_REST_E = config['V_REST_E']
    V_REST_I = config['V_REST_I']
    V_TH = config['V_TH']
    STEEPNESS = config['STEEPNESS']
    
    W_EE = config['W_EE']
    W_EI = config['W_EI']
    W_IE = config['W_IE']
    
    NOISE_LEVEL = config['NOISE_LEVEL']
    
    DRIVE_AMPS = config['DRIVE_AMPS']
    DRIVE_STARTS = config['DRIVE_STARTS']
    DRIVE_ENDS = config['DRIVE_ENDS']
    
    DURATION = config['DURATION']
    
    FIG_SIZE = config['FIG_SIZE']
    FONT_SIZE = config['FONT_SIZE']
    
    COLOR_CYCLE = config['COLOR_CYCLE']
    
    # set up nodes and weights
    nodes = [
        {'tau': TAU_E, 'v_rest': V_REST_E, 'threshold': V_TH, 'steepness': STEEPNESS,},
        {'tau': TAU_I, 'v_rest': V_REST_I, 'threshold': V_TH, 'steepness': STEEPNESS,},
    ]
    weights = np.array([
        [W_EE, W_EI],
        [W_IE,  0.0],
    ])
    
    # build network
    ntwk = network.RateBasedModel(nodes, weights)
    ntwk.store_voltages = True
    ntwk.noise_level = NOISE_LEVEL
    
    # set up drive
    drives = np.zeros((DURATION, len(nodes)), dtype=float)
    for drive_amp, drive_start, drive_end in zip(DRIVE_AMPS, DRIVE_STARTS, DRIVE_ENDS):
        drives[drive_start:drive_end, 0] = drive_amp
        
    # run network
    np.random.seed(SEED)
    ntwk.vs = np.array([V_REST_E, V_REST_I])
    for drive in drives:
        ntwk.step(drive)
        
    # make figure
    fig, axs = plt.subplots(3, 1, figsize=FIG_SIZE, sharex=True, tight_layout=True)
    axs_twin = [ax.twinx() for ax in axs[:2]]
    
    axs[0].plot(np.array(ntwk.vs_history)[:, 0], c=COLOR_CYCLE[0], ls='--', lw=2)
    axs_twin[0].plot(np.array(ntwk.rs_history)[:, 0], c=COLOR_CYCLE[0], ls='-', lw=2)
    
    axs[1].plot(np.array(ntwk.vs_history)[:, 1], c=COLOR_CYCLE[1], ls='--', lw=2)
    axs_twin[1].plot(np.array(ntwk.rs_history)[:, 1], c=COLOR_CYCLE[1], ls='-', lw=2)

    axs[2].set_color_cycle(COLOR_CYCLE)
    axs[2].plot(drives, lw=2)
    
    axs_twin[0].set_ylim(0, 1)
    axs_twin[1].set_ylim(0, 1)
    axs[2].set_ylim(0, drives.max() * 1.1)
    
    axs[0].set_title('Excitatory')
    axs[0].set_ylabel('Voltage')
    axs_twin[0].set_ylabel('Firing rate')
    
    axs[1].set_title('Inhibitory')
    axs[1].set_ylabel('Voltage')
    axs_twin[1].set_ylabel('Firing rate')
    
    axs[2].set_title('Drive')
    axs[2].set_ylabel('Drive')
    axs[2].set_xlabel('t')
    
    for ax in list(axs) + list(axs_twin):
        axis_tools.set_fontsize(ax, FONT_SIZE)