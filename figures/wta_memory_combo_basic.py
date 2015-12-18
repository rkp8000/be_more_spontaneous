from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append('/Users/rkp/Dropbox/Repositories')
sys.path.append('/Users/rkp/Dropbox/Repositories/be_more_spontaneous')
from figure_magic import axis_tools
import network
import network_param_gen


def main(config):
    # unpack params from config
    SEED = config['SEED']
    
    TAU = config['TAU']
    TAU_M = config['TAU_M']
    TAU_C = config['TAU_C']
    V_REST = config['V_REST']
    V_REST_C = config['V_REST_C']
    V_TH = config['V_TH']
    STEEPNESS = config['STEEPNESS']
    
    W_IF = config['W_IF']
    W_FS = config['W_FS']
    W_FI = config['W_FI']
    W_FF = config['W_FF']
    W_FC = config['W_FC']
    
    W_MF = config['W_MF']
    W_MM = config['W_MM']
    W_CF = config['W_CF']
    W_CM = config['W_CM']
    
    N_UNITS = config['N_UNITS']
    
    NOISE_LEVEL = config['NOISE_LEVEL']
    S_DRIVE_AMP = config['S_DRIVE_AMP']
    F0_DRIVE_AMP = config['F0_DRIVE_AMP']
    F1_DRIVE_AMP = config['F1_DRIVE_AMP']
    
    T_F0_DRIVE = config['T_F0_DRIVE']
    D_F0_DRIVE = config['D_F0_DRIVE']
    T_F1_DRIVE = config['T_F1_DRIVE']
    D_F1_DRIVE = config['D_F1_DRIVE']
    T2_F1_DRIVE = config['T2_F1_DRIVE']
    D2_F1_DRIVE = config['D2_F1_DRIVE']
    T2_F0_DRIVE = config['T2_F0_DRIVE']
    D2_F0_DRIVE = config['D2_F0_DRIVE']
    T_S_DRIVE = config['T_S_DRIVE']
    
    DURATION = config['DURATION']
    
    FONT_SIZE = config['FONT_SIZE']
    COLOR_CYCLE = config['COLOR_CYCLE']
    
    # generate network nodes and weights using helper function
    nodes, weights = network_param_gen.wta_memory_combo(
        n_units=N_UNITS,
        tau=TAU, tau_m=TAU_M, tau_c=TAU_C, v_rest=V_REST, v_rest_c=V_REST_C, v_th=V_TH, steepness=STEEPNESS,
        w_if=W_IF, w_fs=W_FS, w_fi=W_FI, w_ff=W_FF, w_fc=W_FC, w_mf=W_MF, w_mm=W_MM, w_cf=W_CF, w_cm=W_CM,
    )
    
    # the order of the neurons in this network is:
    # s, i, f, f, ..., f, f, m, c, c, m, c, c, ..., m, c, c, m, c, c
    # there are N_UNITS f neurons, and N_UNITS*(N_UNITS-1)/2 m, c, c groups
    ntwk = network.RateBasedModel(nodes, weights)
    ntwk.noise_level = NOISE_LEVEL
    ntwk.store_voltages = True
    
    # setup network drive
    s_drive = np.zeros((DURATION, 1), dtype=float)
    s_drive[T_S_DRIVE:, 0] = S_DRIVE_AMP
    f_drive = np.zeros((DURATION, N_UNITS), dtype=float)
    f_drive[T_F0_DRIVE:T_F0_DRIVE+D_F0_DRIVE, 0] = F0_DRIVE_AMP
    f_drive[T_F1_DRIVE:T_F1_DRIVE+D_F1_DRIVE, 1] = F1_DRIVE_AMP
    f_drive[T2_F1_DRIVE:T2_F1_DRIVE+D2_F1_DRIVE, 1] = F1_DRIVE_AMP
    f_drive[T2_F0_DRIVE:T2_F0_DRIVE+D2_F0_DRIVE, 0] = F0_DRIVE_AMP
    
    i_drive = np.zeros((DURATION, 1), dtype=float)
    mcc_drive = np.zeros((DURATION, 3*N_UNITS*(N_UNITS-1)/2), dtype=float)
    
    drives = np.concatenate([s_drive, i_drive, f_drive, mcc_drive], axis=1)
    
    # run simulation
    np.random.seed(SEED)
    ntwk.vs = np.array([n['v_rest'] for n in nodes])
    for drive in drives:
        ntwk.step(drive)
    
    # do some things before making figures
    rs = np.array(ntwk.rs_history)
    vs = np.array(ntwk.vs_history)
    
    f_idxs = np.arange(2, 2 + N_UNITS, dtype=int)
    m_idxs = np.arange(2 + N_UNITS, 2 + N_UNITS + 3 * N_UNITS * (N_UNITS - 1) / 2, 3, dtype=int)
    
    fig, axs = plt.subplots(4, 1, figsize=(15, 12), sharex=True, tight_layout=True)
    axs[3].twin = axs[3].twinx()
    for ax in np.concatenate([axs.flatten(), [axs[3].twin]]):
        ax.set_color_cycle(COLOR_CYCLE)
        
    axs[0].plot(rs[:, f_idxs], lw=2)
    axs[1].plot(drives[:, np.concatenate([f_idxs, [0]])], lw=2)
    axs[2].plot(vs[:, m_idxs], lw=2)
    axs[3].plot(rs[:, range(m_idxs[0] + 1, m_idxs[0] + 3)], lw=2, ls='-')
    
    axs[3].twin.plot(vs[:, range(m_idxs[0] + 1, m_idxs[0] + 3)], lw=2, ls='--')
    
    axs[0].set_title('Fast units')
    axs[0].set_ylabel('Firing rate')
    axs[1].set_title('Drive')
    axs[1].set_ylabel('Drive')
    axs[2].set_title('Memory units')
    axs[2].set_ylabel('Voltage')
    axs[3].set_title('Conduit units')
    axs[3].set_ylabel('Firing rate')
    axs[3].twin.set_ylabel('Voltage')
    axs[3].set_xlabel('t')
    
    for ax in np.concatenate([axs.flatten(), [axs[3].twin]]):
        axis_tools.set_fontsize(ax, FONT_SIZE)