from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append('/Users/rkp/Dropbox/Repositories')
sys.path.append('/Users/rkp/Dropbox/Repositories/be_more_spontaneous')
from figure_magic import axis_tools
import network


def main(config):
    # unpack params from config
    SEED = config['SEED']
    
    TAU_I = config['TAU_I']
    TAU_R = config['TAU_R']
    TAU_B = config['TAU_B']
    TAU_A = config['TAU_A']
    V_REST_I = config['V_REST_I']
    V_REST_R = config['V_REST_R']
    V_REST_B = config['V_REST_B']
    V_REST_A = config['V_REST_A']
    STEEPNESS = config['STEEPNESS']
    THRESHOLD = config['THRESHOLD']
    
    W_IB = config['W_IB']
    W_BI = config['W_BI']
    W_BR = config['W_BR']
    W_AB = config['W_AB']
    W_AA = config['W_AA']
    
    NOISE_LEVEL = config['NOISE_LEVEL']
    
    DURATION = config['DURATION']
    
    DRIVE_11_START = config['DRIVE_11_START']
    DRIVE_11_END = config['DRIVE_11_END']
    DRIVE_11_AMP = config['DRIVE_11_AMP']
    
    DRIVE_12_START = config['DRIVE_12_START']
    DRIVE_12_END = config['DRIVE_12_END']
    DRIVE_12_AMP = config['DRIVE_12_AMP']
    
    DRIVE_13_START = config['DRIVE_13_START']
    DRIVE_13_END = config['DRIVE_13_END']
    DRIVE_13_AMP = config['DRIVE_13_AMP']
    
    DRIVE_21_START = config['DRIVE_21_START']
    DRIVE_21_END = config['DRIVE_21_END']
    DRIVE_21_AMP = config['DRIVE_21_AMP']
    
    COLOR_CYCLE = config['COLOR_CYCLE']
    FIG_SIZE = config['FIG_SIZE']
    FONT_SIZE = config['FONT_SIZE']
    
    # set up nodes
    # order: (I, R1, R2, R3, R4, B12, B13, B14, B23, B24, B34, A12, A13, A14, A23, A24, A34)
    nodes = [
        {'tau': TAU_I, 'v_rest': V_REST_I, 'threshold': THRESHOLD, 'steepness': STEEPNESS},  # I
        {'tau': TAU_R, 'v_rest': V_REST_R, 'threshold': THRESHOLD, 'steepness': STEEPNESS},  # R1
        {'tau': TAU_R, 'v_rest': V_REST_R, 'threshold': THRESHOLD, 'steepness': STEEPNESS},  # R2
        {'tau': TAU_R, 'v_rest': V_REST_R, 'threshold': THRESHOLD, 'steepness': STEEPNESS},  # R3
        {'tau': TAU_R, 'v_rest': V_REST_R, 'threshold': THRESHOLD, 'steepness': STEEPNESS},  # R4
        {'tau': TAU_B, 'v_rest': V_REST_B, 'threshold': THRESHOLD, 'steepness': STEEPNESS},  # B12
        {'tau': TAU_B, 'v_rest': V_REST_B, 'threshold': THRESHOLD, 'steepness': STEEPNESS},  # B13
        {'tau': TAU_B, 'v_rest': V_REST_B, 'threshold': THRESHOLD, 'steepness': STEEPNESS},  # B14
        {'tau': TAU_B, 'v_rest': V_REST_B, 'threshold': THRESHOLD, 'steepness': STEEPNESS},  # B23
        {'tau': TAU_B, 'v_rest': V_REST_B, 'threshold': THRESHOLD, 'steepness': STEEPNESS},  # B24
        {'tau': TAU_B, 'v_rest': V_REST_B, 'threshold': THRESHOLD, 'steepness': STEEPNESS},  # B34
        {'tau': TAU_A, 'v_rest': V_REST_A, 'threshold': THRESHOLD, 'steepness': STEEPNESS},  # A12
        {'tau': TAU_A, 'v_rest': V_REST_A, 'threshold': THRESHOLD, 'steepness': STEEPNESS},  # A13
        {'tau': TAU_A, 'v_rest': V_REST_A, 'threshold': THRESHOLD, 'steepness': STEEPNESS},  # A14
        {'tau': TAU_A, 'v_rest': V_REST_A, 'threshold': THRESHOLD, 'steepness': STEEPNESS},  # A23
        {'tau': TAU_A, 'v_rest': V_REST_A, 'threshold': THRESHOLD, 'steepness': STEEPNESS},  # A24
        {'tau': TAU_A, 'v_rest': V_REST_A, 'threshold': THRESHOLD, 'steepness': STEEPNESS},  # A34
    ]
    
    # set up weight matrix
    weights = np.array([
        #  I    R1    R2    R3    R4    B12   B13   B14   B23   B24   B34   A12   A13   A14   A23   A24   A34
        [ 0.0,  0.0,  0.0,  0.0,  0.0, W_IB, W_IB, W_IB,  W_IB, W_IB, W_IB, 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,],  # I
        [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,],  # R1
        [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,],  # R2
        [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,],  # R3
        [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,],  # R4
        [W_BI, W_BR, W_BR,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,],  # B12
        [W_BI, W_BR,  0.0, W_BR,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,],  # B13
        [W_BI, W_BR,  0.0,  0.0, W_BR,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,],  # B14
        [W_BI,  0.0, W_BR, W_BR,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,],  # B23
        [W_BI,  0.0, W_BR,  0.0, W_BR,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,],  # B24
        [W_BI,  0.0,  0.0, W_BR, W_BR,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,],  # B34
        [ 0.0,  0.0,  0.0,  0.0,  0.0, W_AB,  0.0,  0.0,  0.0,  0.0,  0.0, W_AA,  0.0,  0.0,  0.0,  0.0,  0.0,],  # A12
        [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0, W_AB,  0.0,  0.0,  0.0,  0.0,  0.0, W_AA,  0.0,  0.0,  0.0,  0.0,],  # A13
        [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, W_AB,  0.0,  0.0,  0.0,  0.0,  0.0, W_AA,  0.0,  0.0,  0.0,],  # A14
        [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, W_AB,  0.0,  0.0,  0.0,  0.0,  0.0, W_AA,  0.0,  0.0,],  # A23
        [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, W_AB,  0.0,  0.0,  0.0,  0.0,  0.0, W_AA,  0.0,],  # A24
        [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, W_AB,  0.0,  0.0,  0.0,  0.0,  0.0, W_AA,],  # A34
        #  I    R1    R2    R3    R4    B12   B13   B14   B23   B24   B34   A12   A13   A14   A23   A24   A34
    ])
    
    # build network
    ntwk = network.RateBasedModel(nodes, weights)
    ntwk.store_voltages = True
    ntwk.noise_level = NOISE_LEVEL
    
    # set up drive
    drives = np.zeros((DURATION, len(nodes)), dtype=float)
    drives[DRIVE_11_START:DRIVE_11_END, 1] = DRIVE_11_AMP
    drives[DRIVE_12_START:DRIVE_12_END, 1] = DRIVE_12_AMP
    drives[DRIVE_13_START:DRIVE_13_END, 1] = DRIVE_13_AMP
    drives[DRIVE_21_START:DRIVE_21_END, 2] = DRIVE_21_AMP
    
    # run network
    np.random.seed(SEED)
    ntwk.vs = np.array([node['v_rest'] for node in nodes])
    for drive in drives:
        ntwk.step(drive)


    # open a figure
    fig, axs = plt.subplots(4, 1, figsize=FIG_SIZE, sharex=True, tight_layout=True)
    axs_twinx = [None] + [ax.twinx() for ax in axs[1:]]
    
    # plot drive
    axs[0].set_color_cycle(COLOR_CYCLE)
    axs[0].plot(drives[:, 1:5], lw=2)
    
    # plot responses of receptive field (R) units
    axs[1].set_color_cycle(COLOR_CYCLE)
    axs_twinx[1].set_color_cycle(COLOR_CYCLE)
    axs[1].plot(np.array(ntwk.vs_history)[:, 1:5], ls='--', lw=2)
    axs_twinx[1].plot(np.array(ntwk.rs_history)[:, 1:5], ls='-', lw=2)
    
    # plot responses of buffer (B) units
    axs[2].set_color_cycle(COLOR_CYCLE[4:])
    axs_twinx[2].set_color_cycle(COLOR_CYCLE[4:])
    axs[2].plot(np.array(ntwk.vs_history)[:, 5:11], ls='--', lw=2)
    axs_twinx[2].plot(np.array(ntwk.rs_history)[:,5:11], ls='-', lw=2)
    
    # plot response of inhibitory (I) unit
    axs[2].plot(np.array(ntwk.vs_history)[:, 0], ls='--', lw=2, color='k')
    axs_twinx[2].plot(np.array(ntwk.rs_history)[:, 0], ls='-', lw=2, color='k')
    
    # plot response of association (A) units
    axs[3].set_color_cycle(COLOR_CYCLE[4:])
    axs_twinx[3].set_color_cycle(COLOR_CYCLE[4:])
    axs[3].plot(np.array(ntwk.vs_history)[:, 11:], ls='--', lw=2)
    axs_twinx[3].plot(np.array(ntwk.rs_history)[:, 11:], ls='-', lw=2)
    
    # set axis limits
    for ax in axs_twinx[1:]:
        ax.set_ylim(0, 1)
        
    # label things
    axs[0].set_title('drive')
    axs[0].set_ylabel('strength')
    
    axs[1].set_title('receptive field units')
    axs[1].set_ylabel('voltage')
    axs_twinx[1].set_ylabel('firing rate')
    
    axs[2].set_title('buffer units')
    axs[2].set_ylabel('voltage')
    axs_twinx[2].set_ylabel('firing rate')
    
    axs[3].set_title('association units')
    axs[3].set_ylabel('voltage')
    axs_twinx[3].set_ylabel('firing rate')
    
    axs[3].set_xlabel('time')
    
    for ax in list(axs) + axs_twinx[1:]:
        axis_tools.set_fontsize(ax, FONT_SIZE)