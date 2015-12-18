from __future__ import division, print_function
import numpy as np
import unittest

import network_param_gen


class WtaMemoryComboTestCase(unittest.TestCase):
    
    def test_that_example_networks_get_made_correctly(self):
        # node parameters
        tau = 1
        tau_m = 2
        tau_c = 3
        v_th = 4
        steepness = 5
        v_rest = 6
        v_rest_c = 7
        
        # weight matrix parameters
        w_if = 1  # to inhibitory from fast
        w_fs = 2  # to fast from switch
        w_fi = 3  # to fast from inhibitory
        w_ff = 4  # to fast from fast
        w_fc = 5  # to fast from conduit
        w_mf = 6  # to memory from fast
        w_mm = 7  # to memory from memory
        w_cf = 8  # to conduit from fast
        w_cm = 9  # to conduit from memory
        
        # number of units
        n_units = 3
        
        target_nodes = [
            {'tau': tau, 'v_rest': v_rest, 'threshold': v_th, 'steepness': steepness},    # s
            {'tau': tau, 'v_rest': v_rest, 'threshold': v_th, 'steepness': steepness},    # i
        ] + n_units * [
            {'tau': tau, 'v_rest': v_rest, 'threshold': v_th, 'steepness': steepness},    # f
        ] + int(n_units * (n_units - 1) / 2) * [
            {'tau': tau_m, 'v_rest': v_rest, 'threshold': v_th, 'steepness': steepness},  # m
            {'tau': tau_c, 'v_rest': v_rest_c, 'threshold': v_th, 'steepness': steepness},  # c
            {'tau': tau_c, 'v_rest': v_rest_c, 'threshold': v_th, 'steepness': steepness},  # c
        ]
        
        # first column is s (switch unit)
        # second column is i (inhibitory unit)
        # next n_units columns are f (fast unit)
        # next triplets of 3 columns are m, c, c (memory unit, conduit unit, conduit unit)
        # there is one triplet for every pair of fast units
        # the ordering of the triplets is:
        # (0, 1), (0, 2), ..., (0, n_units-1), (1, 2), (1, 3), ..., (1, n_units-1), ...
        # within each triplet, the first element is the memory unit, the second element is the
        # conduit unit to the lower number from the higher number, and the third element is the
        # conduit unit to the higher number from the lower number
        target_weights = np.array([
            #   m     i    f0    f1    f2  # m01   c01   c10 # m02   c02   c20 # m12   c12   c21 #
            [   0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,],  # s
            [   0,    0, w_if, w_if, w_if,    0,    0,    0,    0,    0,    0,    0,    0,    0,],  # i
            [w_fs, w_fi, w_ff,    0,    0,    0, w_fc,    0,    0, w_fc,    0,    0,    0,    0,],  # f0
            [w_fs, w_fi,    0, w_ff,    0,    0,    0, w_fc,    0,    0,    0,    0, w_fc,    0,],  # f1
            [w_fs, w_fi,    0,    0, w_ff,    0,    0,    0,    0,    0, w_fc,    0,    0, w_fc,],  # f2
            [   0,    0, w_mf, w_mf,    0, w_mm,    0,    0,    0,    0,    0,    0,    0,    0,],  # m01
            [   0,    0,    0, w_cf,    0, w_cm,    0,    0,    0,    0,    0,    0,    0,    0,],  # c01
            [   0,    0, w_cf,    0,    0, w_cm,    0,    0,    0,    0,    0,    0,    0,    0,],  # c10
            [   0,    0, w_mf,    0, w_mf,    0,    0,    0, w_mm,    0,    0,    0,    0,    0,],  # m02
            [   0,    0,    0,    0, w_cf,    0,    0,    0, w_cm,    0,    0,    0,    0,    0,],  # c02
            [   0,    0, w_cf,    0,    0,    0,    0,    0, w_cm,    0,    0,    0,    0,    0,],  # c20
            [   0,    0,    0, w_mf, w_mf,    0,    0,    0,    0,    0,    0, w_mm,    0,    0,],  # m12
            [   0,    0,    0,    0, w_cf,    0,    0,    0,    0,    0,    0, w_cm,    0,    0,],  # c12
            [   0,    0,    0, w_cf,    0,    0,    0,    0,    0,    0,    0, w_cm,    0,    0,],  # c21
            #   m     i    f0    f1    f2  # m01   c01   c10 # m02   c02   c20 # m12   c12   c21 #
        ]).astype(float)
        
        nodes, weights = network_param_gen.wta_memory_combo(
            n_units=n_units,
            tau=tau, tau_m=tau_m, tau_c=tau_c, v_rest=v_rest, v_rest_c=v_rest_c, v_th=v_th, steepness=steepness,
            w_if=w_if, w_fs=w_fs, w_fi=w_fi, w_ff=w_ff, w_fc=w_fc,
            w_mf=w_mf, w_mm=w_mm, w_cf=w_cf, w_cm=w_cm,
        )
        
        self.assertEqual(len(nodes), len(target_nodes))
        self.assertEqual(nodes, target_nodes)
        np.testing.assert_array_equal(weights, target_weights)
        
        # make another one with more units, and make sure size comes out to be correct
        n_units_large = 15
        nodes, weights = network_param_gen.wta_memory_combo(
            n_units=n_units_large,
            tau=tau, tau_m=tau_m, tau_c=tau_c, v_rest=v_rest, v_rest_c=v_rest_c, v_th=v_th, steepness=steepness,
            w_if=w_if, w_fs=w_fs, w_fi=w_fi, w_ff=w_ff, w_fc=w_fc,
            w_mf=w_mf, w_mm=w_mm, w_cf=w_cf, w_cm=w_cm,
        )
        
        target_len = 2 + n_units_large + 3*n_units_large*(n_units_large-1)/2
        self.assertEqual(len(nodes), target_len)
        self.assertEqual(weights.shape, (target_len, target_len))