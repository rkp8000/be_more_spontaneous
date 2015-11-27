from __future__ import division, print_function
import numpy as np


def phi(z):
    """
    Sigmoid function.
    """
    return 1 / (1 + np.exp(-z))


class RateBasedModel(object):
    """
    Rate-based network model. Rates are calculate by passing a population's voltage through a sigmoid 
    function parameterized by a threshold and steepness.
    
    :param nodes: list of nodes -- each node is a dictionary of that node's parameters -- the keys are:
        'tau': time constant
        'baseline': baseline current
        'threshold': firing rate threshold
        'steepness': firing rate steepness
    :param weights: weight matrix, rows are "to", columns are "from", e.g., element at (0, 5) is strength
        connection from node 5 onto node 0
    """
    
    def __init__(self, nodes, weights):
        
        # store all relevant node parameters and connection weight matrix
        self.taus = np.array([node['tau'] for node in nodes], dtype=float)
        self.v_rests = np.array([node['v_rest'] for node in nodes], dtype=float)
        self.v_ths = np.array([node['threshold'] for node in nodes], dtype=float)
        self.gs = np.array([node['steepness'] for node in nodes], dtype=float)
        
        self.w = weights
        
        # set other things to their initial values
        self.noise_level = 0
        
        self._vs = None
        self.rs = None
        
        self.vs_history = []
        
        self.store_voltages = False
    
    def step(self, drive=0):
        """
        Step forward one instant in time, optionally providing a drive to the network.
        
        :param drive: drive to network, given as scalar or 1-D array
        """
        noise = self.noise_level * np.random.randn(*self.vs.shape)
        dvs = (1/self.taus) * (-self.vs + self.w.dot(self.rs) + noise + self.v_rests + drive)
        self.vs = self.vs + dvs  # TO DO: replace with augmented assignment if it works
        
        self.record_data()
    
    def rate_from_voltage(self, vs):
        """
        Calculate firing rate from voltage.
        """
        return phi(self.gs * (vs - self.v_ths))
    
    def record_data(self):
        if self.store_voltages:
            self.vs_history.append(self.vs)
    
    @property
    def vs(self):
        return self._vs
    
    @vs.setter
    def vs(self, vs):
        self._vs = vs
        self.rs = self.rate_from_voltage(self._vs)
        
    @property
    def rs_history(self):
        return [self.rate_from_voltage(vs) for vs in self.vs_history]