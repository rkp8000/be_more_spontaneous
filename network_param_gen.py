from __future__ import division, print_function
from itertools import combinations
import numpy as np


def wta_memory_combo(n_units, tau, tau_m, v_rest, v_rest_c, v_th, steepness,
                     w_if, w_fs, w_fi, w_ff, w_fc, w_mf, w_mm, w_cf, w_cm):
    """
    Create node list and weight matrix for a winner-take-all network with short-term memory.
    
    :param n_units: the number of fast units in the network
    :tau ...
    :w_cm ...
    
    returns: nodes and weights for making network
    """
    # make some useful indices
    fast_idxs = np.arange(2, 2 + n_units, dtype=int)
    m_idxs = np.arange(2 + n_units, 2 + n_units + 3 * n_units * (n_units - 1) / 2, 3)
    pairs = list(combinations(fast_idxs, r=2))
    pairs_inter = [(None, None)] * (2 + n_units)
    for pair in pairs:
        pairs_inter.append((None, None))
        pairs_inter.append(pair)
        pairs_inter.append(pair[::-1])
    
    # make nodes
    nodes = []
    # make switch unit
    nodes.append({'tau': tau, 'v_rest': v_rest, 'threshold': v_th, 'steepness': steepness})
    # make inhibitory unit
    nodes.append({'tau': tau, 'v_rest': v_rest, 'threshold': v_th, 'steepness': steepness})
    # make fast units
    for _ in range(n_units):
        nodes.append({'tau': tau, 'v_rest': v_rest, 'threshold': v_th, 'steepness': steepness})
    # make memory unit triplets
    for pair in pairs:
        nodes.append({'tau': tau_m, 'v_rest': v_rest, 'threshold': v_th, 'steepness': steepness})
        nodes.append({'tau': tau, 'v_rest': v_rest_c, 'threshold': v_th, 'steepness': steepness})
        nodes.append({'tau': tau, 'v_rest': v_rest_c, 'threshold': v_th, 'steepness': steepness})
    
    # make weight matrix
    weights = np.zeros((len(nodes), len(nodes)), dtype=float)
    
    # first row is all zeros
    # next row is input to inhibitory unit, which should just be from fast units
    weights[1, fast_idxs] = w_if
    
    # next n_units rows are fast units
    for idx in fast_idxs:
        # each fast unit receives input from:
        # switch
        weights[idx, 0] = w_fs
        # inhibitory
        weights[idx, 1] = w_fi
        # itself
        weights[idx, idx] = w_ff
        # all the conduits that connect to it
        weights[idx, np.array([pair[0] == idx for pair in pairs_inter], dtype=bool)] = w_fc
                
    # next set of rows are memory-conduit triplets
    for m_idx, pair in zip(m_idxs, pairs):
        
        # fill in memory units
        weights[m_idx, pair] = w_mf
        weights[m_idx, m_idx] = w_mm
        
        # fill in conduit units
        weights[m_idx + 1, pair[1]] = w_cf
        weights[m_idx + 2, pair[0]] = w_cf
        weights[m_idx + 1, m_idx] = w_cm
        weights[m_idx + 2, m_idx] = w_cm
        
    return nodes, weights