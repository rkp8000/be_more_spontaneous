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
    W_XFI = config['W_XFI']
    W_IF = config['W_IF']
    
    SWITCH_DRIVE = config['SWITCH_DRIVE']
    
    FONT_SIZE = config['FONT_SIZE']
    
    ## run simulation with single WTA unit
    # setup network parameters; order: (switch, fast, fast_inh)
    w_single = np.array([
        [0, 0, 0],  # to switch
        [W_FS, W_FF, W_FI],  # to fast
        [0, W_IF, 0],  # to fast_inh
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
    
    axs[0].set_title('Single WTA Unit')
    
    for ax in axs:
        axis_tools.set_fontsize(ax, FONT_SIZE)
        
    ## run simulation with three non-interacting WTA units
    # setup network parameters; order: (switch, fast, fast_inh, fast, fast_inh, fast, fast_inh)
    w_multi = np.array([
        [0, 0, 0, 0, 0, 0, 0],  # to switch
        [W_FS, W_FF, W_FI, 0, 0, 0, 0],  # to fast1
        [0, W_IF, 0, 0, 0, 0, 0],  # to fast_inh1
        [W_FS, 0, 0, W_FF, W_FI, 0, 0],  # to fast2
        [0, 0, 0, W_IF, 0, 0, 0],  # to fast_inh2
        [W_FS, 0, 0, 0, 0, W_FF, W_FI],  # to fast3
        [0, 0, 0, 0, 0, W_IF, 0],  # to fast_inh3
    ])
    nodes_multi = 7 * [{'tau': TAU, 'v_rest': V_REST, 'threshold': THRESHOLD, 'steepness': STEEPNESS}]
    
    # build network
    ntwk = network.RateBasedModel(nodes_multi, w_multi)
    ntwk.noise_level = NOISE_LEVEL
    ntwk.store_voltages = True
    
    # set network drive
    drive = np.array([SWITCH_DRIVE, 0, 0, 0, 0, 0, 0])
    
    # run network
    np.random.seed(SEED)
    ntwk.vs = np.array([V_REST, V_REST, V_REST, V_REST, V_REST, V_REST, V_REST])
    for t in range(DURATION):
        ntwk.step(drive)

    # make figure
    fig, axs = plt.subplots(2, 1, figsize=(15, 5), sharex=True, tight_layout=True)
    axs[0].plot(np.array(ntwk.vs_history)[:, [1, 3, 5]], lw=2)
    axs[1].plot(np.array(ntwk.rs_history)[:, [1, 3, 5]], lw=2)
    
    axs[1].set_ylim(-.1, 1.1)
    axs[1].set_yticks(np.linspace(0, 1, 5, endpoint=True))
    axs[0].set_ylabel('Voltage')
    axs[1].set_xlabel('t')
    axs[1].set_ylabel('Firing rate')
    
    axs[0].set_title('Three Independent WTA Units')
    
    for ax in axs:
        axis_tools.set_fontsize(ax, FONT_SIZE)
        
        
    ## run simulation with three interacting WTA units
    # setup network parameters; order: (switch, fast1, fast_inh1, fast2, fast_inh2, fast3, fast_inh3)
    w_multi = np.array([
        [0, 0, 0, 0, 0, 0, 0],  # to switch
        [W_FS, W_FF, W_FI, 0, W_XFI, 0, W_XFI],  # to fast1
        [0, W_IF, 0, 0, 0, 0, 0],  # to fast_inh1
        [W_FS, 0, W_XFI, W_FF, W_FI, 0, W_XFI],  # to fast2
        [0, 0, 0, W_IF, 0, 0, 0],  # to fast_inh2
        [W_FS, 0, W_XFI, 0, W_XFI, W_FF, W_FI],  # to fast3
        [0, 0, 0, 0, 0, W_IF, 0],  # to fast_inh3
    ])
    nodes_multi = 7 * [{'tau': TAU, 'v_rest': V_REST, 'threshold': THRESHOLD, 'steepness': STEEPNESS}]
    
    # build network
    ntwk = network.RateBasedModel(nodes_multi, w_multi)
    ntwk.noise_level = NOISE_LEVEL
    ntwk.store_voltages = True
    
    # set network drive
    drive = np.array([SWITCH_DRIVE, 0, 0, 0, 0, 0, 0])
    
    # run network
    np.random.seed(SEED)
    ntwk.vs = np.array([V_REST, V_REST, V_REST, V_REST, V_REST, V_REST, V_REST])
    for t in range(DURATION):
        ntwk.step(drive)

    # make figure
    fig, axs = plt.subplots(2, 1, figsize=(15, 5), sharex=True, tight_layout=True)
    axs[0].plot(np.array(ntwk.vs_history)[:, [1, 3, 5]], lw=2)
    axs[1].plot(np.array(ntwk.rs_history)[:, [1, 3, 5]], lw=2)
    
    axs[1].set_ylim(-.1, 1.1)
    axs[1].set_yticks(np.linspace(0, 1, 5, endpoint=True))
    axs[0].set_ylabel('Voltage')
    axs[1].set_xlabel('t')
    axs[1].set_ylabel('Firing rate')
    
    axs[0].set_title('Three Interacting WTA Units')
    
    for ax in axs:
        axis_tools.set_fontsize(ax, FONT_SIZE)
    
    
    ## run simulation with three fast units all of whose interactions occur through one inhibitory unit
    # setup network parameters; order: (switch, inh, fast1, fast2, fast3)
    w_multi = np.array([
        [0, 0, 0, 0, 0],  # to switch
        [0, 0, W_IF, W_IF, W_IF],  # to inh
        [W_FS, W_FI, W_FF, 0, 0],  # to fast1
        [W_FS, W_FI, 0, W_FF, 0],  # to fast2
        [W_FS, W_FI, 0, 0, W_FF],  # to fast3
    ])
    nodes_multi = 5 * [{'tau': TAU, 'v_rest': V_REST, 'threshold': THRESHOLD, 'steepness': STEEPNESS}]
    
    # build network
    ntwk = network.RateBasedModel(nodes_multi, w_multi)
    ntwk.noise_level = NOISE_LEVEL
    ntwk.store_voltages = True
    
    # set network drive
    drive = np.array([SWITCH_DRIVE, 0, 0, 0, 0])
    
    # run network
    np.random.seed(SEED)
    ntwk.vs = np.array([V_REST, V_REST, V_REST, V_REST, V_REST])
    for t in range(DURATION):
        ntwk.step(drive)

    # make figure
    fig, axs = plt.subplots(2, 1, figsize=(15, 5), sharex=True, tight_layout=True)
    axs[0].plot(np.array(ntwk.vs_history)[:, 2:], lw=2)
    axs[1].plot(np.array(ntwk.rs_history)[:, 2:], lw=2)
    
    axs[1].set_ylim(-.1, 1.1)
    axs[1].set_yticks(np.linspace(0, 1, 5, endpoint=True))
    axs[0].set_ylabel('Voltage')
    axs[1].set_xlabel('t')
    axs[1].set_ylabel('Firing rate')
    
    axs[0].set_title('Three Exc Units Sharing One Inh Unit')
    
    for ax in axs:
        axis_tools.set_fontsize(ax, FONT_SIZE)
    
    ## run simulation with many fast units all of whose interactions occur through one inhibitory unit
    # setup network parameters; order: (switch, inh, fast1, fast2, fast3)
    w_multi = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # to switch
        [0, 0, W_IF, W_IF, W_IF, W_IF, W_IF, W_IF, W_IF, W_IF],  # to inh
        [W_FS, W_FI, W_FF, 0, 0, 0, 0, 0, 0, 0],  # to fast1
        [W_FS, W_FI, 0, W_FF, 0, 0, 0, 0, 0, 0],  # to fast2
        [W_FS, W_FI, 0, 0, W_FF, 0, 0, 0, 0, 0],  # to fast3
        [W_FS, W_FI, 0, 0, 0, W_FF, 0, 0, 0, 0],  # to fast3
        [W_FS, W_FI, 0, 0, 0, 0, W_FF, 0, 0, 0],  # to fast3
        [W_FS, W_FI, 0, 0, 0, 0, 0, W_FF, 0, 0],  # to fast3
        [W_FS, W_FI, 0, 0, 0, 0, 0, 0, W_FF, 0],  # to fast3
        [W_FS, W_FI, 0, 0, 0, 0, 0, 0, 0, W_FF],  # to fast3
    ])
    nodes_multi = 10 * [{'tau': TAU, 'v_rest': V_REST, 'threshold': THRESHOLD, 'steepness': STEEPNESS}]
    
    # build network
    ntwk = network.RateBasedModel(nodes_multi, w_multi)
    ntwk.noise_level = NOISE_LEVEL
    ntwk.store_voltages = True
    
    # set network drive
    drive = np.array([SWITCH_DRIVE, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    
    # run network
    np.random.seed(SEED)
    ntwk.vs = np.array([V_REST, V_REST, V_REST, V_REST, V_REST, V_REST, V_REST, V_REST, V_REST, V_REST])
    for t in range(DURATION):
        ntwk.step(drive)

    # make figure
    fig, axs = plt.subplots(2, 1, figsize=(15, 5), sharex=True, tight_layout=True)
    axs[0].plot(np.array(ntwk.vs_history)[:, 2:], lw=2)
    axs[1].plot(np.array(ntwk.rs_history)[:, 2:], lw=2)
    
    axs[1].set_ylim(-.1, 1.1)
    axs[1].set_yticks(np.linspace(0, 1, 5, endpoint=True))
    axs[0].set_ylabel('Voltage')
    axs[1].set_xlabel('t')
    axs[1].set_ylabel('Firing rate')
    
    axs[0].set_title('Eight Exc Units Sharing One Inh Unit')
    
    for ax in axs:
        axis_tools.set_fontsize(ax, FONT_SIZE)
        