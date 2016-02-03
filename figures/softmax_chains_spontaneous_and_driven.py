from __future__ import division, print_function
from copy import deepcopy as copy
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import axis_tools
import network
import network_param_gen


def spontaneous(config):
    """
    Spontaneous dynamics simulation.
    """
    
    # parse config parameters
    SEED = config['SEED']
    
    N_CHAINS = config['N_CHAINS']
    CHAIN_LENGTH = config['CHAIN_LENGTH']
    GAIN = config['GAIN']
    
    DURATION = config['DURATION']
    
    N_TRIALS = config['N_TRIALS']
    
    FIG_SIZE = config['FIG_SIZE']
    COLORS = config['COLORS']
    LW = config['LW']
    FONT_SIZE = config['FONT_SIZE']
    
    # make base network
    weights = 2 * network_param_gen.chain_weight_matrix(
        n_chains=N_CHAINS, chain_length=CHAIN_LENGTH,
    )
    ntwk_base = network.RecurrentSoftMaxModel(
        weights=weights, gain=GAIN, shape=(N_CHAINS, CHAIN_LENGTH)
    )
    ntwk_base.store_voltages = True
    
    # run networks and store output firing rates
    rs_plottables = []
    np.random.seed(SEED)
    for _ in range(N_TRIALS):
        
        ntwk = copy(ntwk_base)
        
        # run network with no drive
        ntwk.vs = np.zeros((N_CHAINS * CHAIN_LENGTH,), dtype=float)
        for t in range(DURATION):
            ntwk.step()
    
        # do a bit of reshaping on responses to get plottable arrays
        rs_matrix = np.array(
            [r.reshape(N_CHAINS, CHAIN_LENGTH) for r in ntwk.rs_history]
        )
        rs_plottables.append([rs_matrix[:, :, chain_pos] for chain_pos in range(CHAIN_LENGTH)])
    
    # plot activity history for all trials
    fig, axs = plt.subplots(
        CHAIN_LENGTH, N_TRIALS, figsize=FIG_SIZE,
        sharex=True, sharey=True, tight_layout=True
    )
    
    for t_ctr in range(N_TRIALS):
        
        for ctr, (rs, ax) in enumerate(zip(rs_plottables[t_ctr], axs[:, t_ctr])):
            
            if ctr == 0:
                ax.set_title('Trial {}'.format(t_ctr))
                
            ax.set_color_cycle(COLORS)
            ax.plot(rs, lw=LW)
            
    for ax in axs[:, 0]:
        ax.set_ylabel('Rate')
    
    for ax in axs[-1, :]:
        ax.set_xlabel('Time')
    
    for ax in axs.flatten():
        ax.set_xlim(0, DURATION - 1)
        ax.set_ylim(0, 1)
        axis_tools.set_fontsize(ax, FONT_SIZE)
        

def driven(config):
    """
    Driven dynamics simulation.
    """
    
    # parse config parameters
    SEED = config['SEED']
    
    N_CHAINS = config['N_CHAINS']
    CHAIN_LENGTH = config['CHAIN_LENGTH']
    GAIN = config['GAIN']
    
    DRIVE_NODE_COORDINATES = config['DRIVE_NODE_COORDINATES']
    DRIVE_AMPLITUDE = config['DRIVE_AMPLITUDE']
    
    N_TRIALS = config['N_TRIALS']
    
    FIG_SIZE = config['FIG_SIZE']
    COLORS = config['COLORS']
    LW = config['LW']
    FONT_SIZE = config['FONT_SIZE']
    
    # make base network
    weights = 2 * network_param_gen.chain_weight_matrix(
        n_chains=N_CHAINS, chain_length=CHAIN_LENGTH,
    )
    ntwk_base = network.RecurrentSoftMaxModel(
        weights=weights, gain=GAIN, shape=(N_CHAINS, CHAIN_LENGTH),
    )
    ntwk_base.store_voltages = True
    
    # run networks and store output firing rates
    drives_plottables = []
    rs_plottables = []
    np.random.seed(SEED)
    
    for t_ctr in range(N_TRIALS):
        
        ntwk = copy(ntwk_base)
        
        ntwk.vs = np.zeros((N_CHAINS * CHAIN_LENGTH,), dtype=float)
        
        drives_matrix = []
        
        for node_coord in DRIVE_NODE_COORDINATES[t_ctr]:
            drive = np.zeros((N_CHAINS, CHAIN_LENGTH), dtype=float)
            drive[node_coord] = DRIVE_AMPLITUDE
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
        ax.set_xlim(0, len(DRIVE_NODE_COORDINATES[0]) - 1)
        ax.set_ylim(0, 1)
        axis_tools.set_fontsize(ax, FONT_SIZE)