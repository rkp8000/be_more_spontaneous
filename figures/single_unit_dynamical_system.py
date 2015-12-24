from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append('/Users/rkp/Dropbox/Repositories')
sys.path.append('/Users/rkp/Dropbox/Repositories/be_more_spontaneous')
from figure_magic import axis_tools


def phi(z):
    return 1 / (1 + np.exp(-z))


def main(config):
    TAU_F = config['TAU_F']
    W_FF = config['W_FF']
    V_TH = config['V_TH']
    G = config['G']
    V_0 = config['V_0']
    DS = config['DS']
    
    FONT_SIZE = config['FONT_SIZE']
    FIG_SIZE = config['FIG_SIZE']
    
    def f(v, d):
        return (1 / TAU_F) * (-v + W_FF*phi(G*(v - V_TH)) + V_0 + d)
    
    vs = np.linspace(-6, 10, 200)
    fs = [f(vs, d) for d in DS]

    fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE)
    ax.set_xlabel('v', fontsize=16)
    ax.set_ylabel('f(v)', fontsize=16)
    ax.plot(vs, np.transpose(fs), lw=2)
    ax.legend(['d = {}'.format(d) for d in DS], loc='best')
    ax.axvline(0, ls='--', lw=1, c='k')
    ax.axhline(0, ls='--', lw=1, c='k')
    
    axis_tools.set_fontsize(ax, FONT_SIZE)