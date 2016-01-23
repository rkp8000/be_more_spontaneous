from __future__ import division, print_function
from itertools import product as cproduct
import matplotlib.pyplot as plt
import numpy as np
import sys
import unittest

sys.path.append('/Users/rkp/Dropbox/Repositories')
from figure_magic import axis_tools
from unittest_jupyter import run

from network import RateBasedModel, DiscreteTimeSquareLattice


FONT_SIZE = 20

class RateModelNetworkTestCase(unittest.TestCase):
    
    def test0_unconnected_populations_with_different_timescales_respond_to_delta_pulse_with_exponential_decay(self):
        # define network drive
        drives = 10*[np.array([0, 0, 0, 0])] + 10*[2*np.array([3, 4, 6, 10])] + 60*[np.array([0, 0, 0, 0])]
        # define network parameters
        taus = [3, 5, 10, 20]
        v_rests = [-20, -10, 0, 10]
        v_ths = [-10, -1, 8, 17]
        gs = [.5, .5, .5, .5]
        
        # dake new network
        nodes = [{'tau': t, 'v_rest': v_rest, 'threshold': th, 'steepness': g}
                for t, v_rest, th, g in zip(taus, v_rests, v_ths, gs)]
        weights = np.zeros((len(taus), len(taus)))
        
        ntwk = RateBasedModel(nodes, weights)
        ntwk.store_voltages = True
        
        # initialize and run network
        ntwk.vs = np.array(v_rests)
        for drive in drives:
            ntwk.step(drive)
        
        # make plots
        fig, axs = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
        axs[0].plot(ntwk.vs_history, lw=2)
        axs[1].plot(ntwk.rs_history, lw=2)
        axs[2].plot(drives, lw=2)
        
        axs[1].set_ylim(-.1, 1.1)
        axs[2].set_ylim(0, 45)
        
        axs[0].set_ylabel('voltage')
        axs[1].set_ylabel('firing rate')
        axs[2].set_ylabel('drive')
        axs[2].set_xlabel('t')
        
        axs[0].set_title('Uncoupled Populations')
        
        for ax in axs:
            axis_tools.set_fontsize(ax, 20)
            
    def test1_self_connections_yield_bistable_systems_given_correct_parameters(self):
        # define network drive
        drives = 10*[np.array([0, 0])] + 20*[4*np.array([1, 1])] + \
                 60*[np.array([0, 0])] + 20*[-4*np.array([1, 1])] + 60*[np.array([0, 0])]
        # define network parameters
        taus = [10, 10]
        v_rests = [0, 0]
        v_ths = [4, 4]
        gs = [2.5, 2.5]
        w_selfs = [7.5, 5.5]
        
        nodes = [{'tau': t, 'v_rest': v_rest, 'threshold': th, 'steepness': g}
                for t, v_rest, th, g in zip(taus, v_rests, v_ths, gs)]
        weights = np.diag(w_selfs)
        
        # make network
        ntwk = RateBasedModel(nodes, weights)
        ntwk.store_voltages = True
        
        # initialize and run network
        ntwk.vs = np.array(v_rests)
        for drive in drives:
            ntwk.step(drive)
            
        # make plots
        fig, axs = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
        axs[0].plot(ntwk.vs_history, lw=2)
        axs[1].plot(ntwk.rs_history, lw=2)
        axs[2].plot(drives, lw=2, c='k')
        
        axs[1].set_ylim(-.1, 1.1)
        axs[2].set_ylim(-5, 5)
        
        axs[0].set_ylabel('voltage')
        axs[1].set_ylabel('firing rate')
        axs[2].set_ylabel('drive')
        axs[2].set_xlabel('t')
        
        axs[0].set_title('Self-Connected Populations')
        
        for ax in axs:
            axis_tools.set_fontsize(ax, 20)
            
    def test2_upstate_timecourse_with_self_connections_depends_on_noise(self):
        # define network drive
        drives = 10*[np.array([0, 0, 0])] + 40*[4*np.array([1, 1, 1])] + \
                 140*[np.array([0, 0, 0])]
        # define network parameters
        taus = [10, 10, 10]
        v_rests = [0, 0, 0]
        v_ths = [4, 4, 4]
        gs = [2.5, 2.5, 2.5]
        w_selfs = [5.3, 5.3, 5.3]
        noise_level = np.array([1, 2, 3])
        
        nodes = [{'tau': t, 'v_rest': v_rest, 'threshold': th, 'steepness': g}
                for t, v_rest, th, g in zip(taus, v_rests, v_ths, gs)]
        weights = np.diag(w_selfs)
        
        # make network
        ntwk = RateBasedModel(nodes, weights)
        ntwk.store_voltages = True
        ntwk.noise_level = noise_level
        
        # initialize and run network
        np.random.seed(seed=4)
        ntwk.vs = np.array(v_rests)
        for drive in drives:
            ntwk.step(drive)
            
        # make plots
        fig, axs = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
        axs[0].plot(ntwk.vs_history, lw=2)
        axs[1].plot(ntwk.rs_history, lw=2)
        axs[2].plot(drives, lw=2, c='k')
        
        axs[1].set_ylim(-.1, 1.1)
        axs[2].set_ylim(-1, 5)
        
        axs[0].set_ylabel('voltage')
        axs[1].set_ylabel('firing rate')
        axs[2].set_ylabel('drive')
        axs[2].set_xlabel('t')
        
        axs[0].set_title('Self-Connected Populations With Noise')
        
        for ax in axs:
            axis_tools.set_fontsize(ax, 20)
            
    def test3_input_populations_coupled_to_bistable_population(self):
        # define network drive
        drive_1 = 10*[0] + 20*[10] + 200*[0]
        drive_2 = 15*[0] + 20*[10] + 195*[0]
        drive_12 = len(drive_1) * [0]
        drives = np.transpose([drive_1, drive_2, drive_12])
        
        # define network parameters
        taus = [10, 10, 10]
        v_rests = [0, 0, 0]
        v_ths = [4, 4, 4]
        gs = [2.5, 2.5, 2.5]
        noise_level = 2
        
        nodes = [{'tau': t, 'v_rest': v_rest, 'threshold': th, 'steepness': g}
                for t, v_rest, th, g in zip(taus, v_rests, v_ths, gs)]
        weights = np.array([
                [0, 0, 0],
                [0, 0, 0],
                [2, 2, 5.5]
            ])
        
        # make network
        ntwk = RateBasedModel(nodes, weights)
        ntwk.store_voltages = True
        ntwk.noise_level = noise_level
        
        # initialize and run network
        np.random.seed(1)
        ntwk.vs = np.array(v_rests)
        for drive in drives:
            ntwk.step(drive)
            
        fig, axs = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
        axs[0].plot(ntwk.vs_history, lw=2)
        axs[1].plot(ntwk.rs_history, lw=2)
        axs[2].plot(drives, lw=2)
        
        axs[1].set_ylim(-.1, 1.1)
        axs[2].set_ylim(-1, 11)
        
        axs[0].set_ylabel('voltage')
        axs[1].set_ylabel('firing rate')
        axs[2].set_ylabel('drive')
        axs[2].set_xlabel('t')
        
        axs[0].set_title('Input populations coupled to bistable population')
        
        for ax in axs:
            axis_tools.set_fontsize(ax, 20)
            

class DiscreteTimeSquareLatticeTestCase(unittest.TestCase):
    
    def test_make_weight_matrix_makes_correct_weight_matrices(self):
        # we'll try out a 4 x 5 lattice for the nearest neighbor setting
        # [x x x x x]
        # [x x x x x]
        # [x x x x x]
        # [x x x x x]
        # to vectorize the lattice we read left to right, top to bottom
        correct_matrix = np.array([
          # 00 01 02 03 04   10 11 12 13 14   20 21 22 23 24  30 31 32 33 34
           [0, 1, 0, 0, 0,   1, 1, 0, 0, 0,   0, 0, 0, 0, 0,  0, 0, 0, 0, 0,],  # 00
           [1, 0, 1, 0, 0,   1, 1, 1, 0, 0,   0, 0, 0, 0, 0,  0, 0, 0, 0, 0,],  # 01
           [0, 1, 0, 1, 0,   0, 1, 1, 1, 0,   0, 0, 0, 0, 0,  0, 0, 0, 0, 0,],  # 02
           [0, 0, 1, 0, 1,   0, 0, 1, 1, 1,   0, 0, 0, 0, 0,  0, 0, 0, 0, 0,],  # 03
           [0, 0, 0, 1, 0,   0, 0, 0, 1, 1,   0, 0, 0, 0, 0,  0, 0, 0, 0, 0,],  # 04
                
           [1, 1, 0, 0, 0,   0, 1, 0, 0, 0,   1, 1, 0, 0, 0,  0, 0, 0, 0, 0,],  # 10
           [1, 1, 1, 0, 0,   1, 0, 1, 0, 0,   1, 1, 1, 0, 0,  0, 0, 0, 0, 0,],  # 11
           [0, 1, 1, 1, 0,   0, 1, 0, 1, 0,   0, 1, 1, 1, 0,  0, 0, 0, 0, 0,],  # 12
           [0, 0, 1, 1, 1,   0, 0, 1, 0, 1,   0, 0, 1, 1, 1,  0, 0, 0, 0, 0,],  # 13
           [0, 0, 0, 1, 1,   0, 0, 0, 1, 0,   0, 0, 0, 1, 1,  0, 0, 0, 0, 0,],  # 14
                
           [0, 0, 0, 0, 0,   1, 1, 0, 0, 0,   0, 1, 0, 0, 0,  1, 1, 0, 0, 0,],  # 20
           [0, 0, 0, 0, 0,   1, 1, 1, 0, 0,   1, 0, 1, 0, 0,  1, 1, 1, 0, 0,],  # 21
           [0, 0, 0, 0, 0,   0, 1, 1, 1, 0,   0, 1, 0, 1, 0,  0, 1, 1, 1, 0,],  # 22
           [0, 0, 0, 0, 0,   0, 0, 1, 1, 1,   0, 0, 1, 0, 1,  0, 0, 1, 1, 1,],  # 23
           [0, 0, 0, 0, 0,   0, 0, 0, 1, 1,   0, 0, 0, 1, 0,  0, 0, 0, 1, 1,],  # 24
                
           [0, 0, 0, 0, 0,   0, 0, 0, 0, 0,   1, 1, 0, 0, 0,  0, 1, 0, 0, 0,],  # 30
           [0, 0, 0, 0, 0,   0, 0, 0, 0, 0,   1, 1, 1, 0, 0,  1, 0, 1, 0, 0,],  # 31
           [0, 0, 0, 0, 0,   0, 0, 0, 0, 0,   0, 1, 1, 1, 0,  0, 1, 0, 1, 0,],  # 32
           [0, 0, 0, 0, 0,   0, 0, 0, 0, 0,   0, 0, 1, 1, 1,  0, 0, 1, 0, 1,],  # 33
           [0, 0, 0, 0, 0,   0, 0, 0, 0, 0,   0, 0, 0, 1, 1,  0, 0, 0, 1, 0,],  # 34
        ], dtype=float)
        shape = (4, 5)
        weight_type = 'nearest_neighbor_diagonal'
        
        np.testing.assert_array_equal(DiscreteTimeSquareLattice.make_weight_matrix(shape, weight_type), correct_matrix)
        
    def test_correct_example_update_works(self):
        # test two networks with different thresholds
        ntwk = DiscreteTimeSquareLattice(
            shape=(5, 8),
            activation_strength=2,
            inactivation_strength=-2,
            threshold=1.5,
            steepness=1000,
            weight_type='nearest_neighbor_diagonal',
        )
        
        ntwk.vs_matrix = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 2, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])
        
        correct_next_vs_matrix = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 2, 2, 2, 0, 0, 0, 0],
            [0, 2, 1,-2, 0, 0, 0, 0],
            [0, 2, 2, 2, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])
        
        ntwk.step()
        np.testing.assert_array_equal(ntwk.vs_matrix, correct_next_vs_matrix)
        
        # second network to test
        ntwk = DiscreteTimeSquareLattice(
            shape=(5, 8),
            activation_strength=2,
            inactivation_strength=-2,
            threshold=2.5,
            steepness=1000,
            weight_type='nearest_neighbor_diagonal',
        )
        
        ntwk.vs_matrix = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 2, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])
        
        correct_next_vs_matrix = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 2, 2, 0, 0, 0, 0],
            [0, 0, 1,-2, 0, 0, 0, 0],
            [0, 0, 2, 2, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])
        
        ntwk.step()
        np.testing.assert_array_equal(ntwk.vs_matrix, correct_next_vs_matrix)
        
        
if __name__ == '__main__':
    unittest.main()