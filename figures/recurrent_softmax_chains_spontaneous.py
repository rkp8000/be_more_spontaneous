from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import axis_tools
import network
import network_param_gen


def main(config):
    # parse config parameters
    SEED = config['SEED']
    
    N_CHAINS = config['N_CHAINS']
    CHAIN_LENGTH = config['CHAIN_LENGTH']
    GAIN = config['GAIN']
    
    DURATION = config['DURATION']
    
    FIG_SIZE = config['FIG_SIZE']
    COLORS = config['COLORS']
    LW = config['LW']
    FONT_SIZE = config['FONT_SIZE']
    
    # make network
    weights = network_param_gen.chain_weight_matrix(
        n_chains=N_CHAINS, chain_length=CHAIN_LENGTH,
    )
    
    ntwk = network.RecurrentSoftMaxModel(weights=weights, gain=GAIN, shape=(N_CHAINS, CHAIN_LENGTH))
    ntwk.store_voltages = True
    
    # run network with no drive
    np.random.seed(SEED)
    ntwk.vs = np.zeros((N_CHAINS * CHAIN_LENGTH,), dtype=float)
    for t in range(DURATION):
        ntwk.step()
    
    # do a bit of reshaping on responses to get plottable arrays
    rs_matrix = np.array(
        [r.reshape(N_CHAINS, CHAIN_LENGTH) for r in ntwk.rs_history]
    )
    rs_plottable = [rs_matrix[:, :, chain_pos] for chain_pos in range(CHAIN_LENGTH)]
    
    # plot activity history
    fig, axs = plt.subplots(CHAIN_LENGTH, 1, figsize=FIG_SIZE, tight_layout=True)
    for ctr, (rs, ax) in enumerate(zip(rs_plottable, axs)):
        ax.set_title('Chain position {}'.format(ctr))
        ax.set_color_cycle(COLORS)
        ax.plot(rs, lw=LW)
    
    for ax in axs:
        ax.set_ylabel('Firing rate')
        
    axs[-1].set_xlabel('t')
    
    for ax in axs:
        axis_tools.set_fontsize(ax, FONT_SIZE)