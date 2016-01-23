from __future__ import division, print_function
import numpy as np


def phi(z):
    """
    Sigmoid function.
    """
    return 1 / (1 + np.exp(-z))


class VoltageFiringRateModel(object):
    """
    Base class for models where each node has a voltage and a firing rate. The conversion function must be 
    specified in the child class in a function called rate_from_voltage. 
    Also, you must call super(self.__class__, self).__init__() in the child class's __init__ method.
    """
    def __init__(self):
        self._vs = None
        self.rs = None
        
        self.vs_history = []
        
        self.store_voltages = False
        
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

    
class RateBasedModel(VoltageFiringRateModel):
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
        
        super(self.__class__, self).__init__()
    
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
    
    
class DiscreteTimeSquareLattice(VoltageFiringRateModel):
    """
    Simple, Conway's-Game-of-Life-esque network arranged on a square lattice. Each unit's activation is determined by the activations of its nearest neighbors.
    
    :param shape: lattice dimensions
    :param activation: value that a node achieves upon activation
    :param inactivation: value that a node achieves upon inactivation
    :param threshold: threshold parameter for activation probability sigmoid
    :param steepness: steepness parameter for activation probability sigmoid
    :param weight_type: how to make the weight matrix; allowed options are:
        'nearest_neighbor_diagonal'
    :param **weight_matrix_kwargs: other parameters for making weight matrices; required params are:
        'nearest_neighbor_diagonal': none
    """
    
    @staticmethod
    def make_weight_matrix(shape, weight_type, **kwargs):
        """Make a lattice-based weight matrix."""
        
        assert weight_type in [
            'nearest_neighbor_diagonal'
        ]
        
        n_nodes = shape[0] * shape[1]
        
        if weight_type == 'nearest_neighbor_diagonal':
            diag_block = np.zeros((shape[1], shape[1]), dtype=float)
            for row in range(shape[1]):
                if row == 0:
                    diag_block[row, row + 1] = 1
                elif row == shape[1] - 1:
                    diag_block[row, row - 1] = 1
                else:
                    diag_block[row, row - 1] = 1
                    diag_block[row, row + 1] = 1
            off_diag_block = np.zeros((shape[1], shape[1]), dtype=float)
            for row in range(shape[1]):
                if row == 0:
                    off_diag_block[row, row] = 1
                    off_diag_block[row, row+1] = 1
                elif row == shape[1] - 1:
                    off_diag_block[row, row-1] = 1
                    off_diag_block[row, row] = 1
                else:
                    off_diag_block[row, row-1] = 1
                    off_diag_block[row, row] = 1
                    off_diag_block[row, row+1] = 1
                    
            big_array = [
                [np.zeros((shape[1], shape[1]), dtype=float) for _ in range(shape[0])]
                for _ in range(shape[0])
            ]
            
            for row in range(shape[0]):
                big_array[row][row] = diag_block.copy()
                if row == 0:
                    big_array[row][row+1] = off_diag_block.copy()
                elif row == shape[0] - 1:
                    big_array[row][row-1] = off_diag_block.copy()
                else:
                    big_array[row][row-1] = off_diag_block.copy()
                    big_array[row][row+1] = off_diag_block.copy()

            temp = [np.concatenate(big_row, axis=1) for big_row in big_array]
            w = np.concatenate(temp, axis=0)
            
        return w
            
    def __init__(self, shape, activation_strength, inactivation_strength, 
                 threshold, steepness, weight_type, **weight_matrix_kwargs):

        self.shape = shape
        self.activation_strength = activation_strength
        self.inactivation_strength = inactivation_strength
        self.threshold = threshold
        self.steepness = steepness
        self.w = self.make_weight_matrix(shape, weight_type, **weight_matrix_kwargs)
        
        self.n_nodes = np.prod(shape)
        
        self.voltages = np.zeros((self.n_nodes,), dtype=float)
        self.firing_rates = np.zeros((self.nodes,), dtype=float)
        self.node_inputs = np.zeros((self.n_nodes,), dtype=float)
        
        super(self.__class__, self).__init__()
    
    def step(self, scalar_drive=0, matrix_drive=None):
        """
        Step forward one instant in time, optionally providing drive to the network.
        
        :param drive: drive to network, given as scalar or 1-D array
        """
        drive = scalar_drive
        if matrix_drive is not None:
            drive += matrix_drive.flatten()
        
        self.node_inputs = self.w.dot(self.rs) + drive
        
        # update all voltages
        new_vs = self.vs.copy()
        
        # decrement all activations greater than 1
        new_vs[self.vs > 1] -= 1
        # increment all negative activations
        new_vs[self.vs < 0] += 1
        # set all smallest activations to self.inactivation_strength
        new_vs[self.vs == 1] = self.inactivation_strength
        
        # probabilistically activate nodes receiving highest input
        self.activation_probabilities = self.sigmoid(self.node_inputs.copy())
        # only allow nodes with zero activation value to have the chance to activate
        self.activation_probabilities[self.vs != 0] = 0
        # sample which nodes have become active, and set their voltage to self.activation_strength
        new_vs[np.random.rand(self.n_nodes) < self.activation_probabilities] = self.activation_strength
        
        # store voltages
        self.vs = vs
        
        self.record_data()
        
    def rate_from_voltage(self, vs):
        """
        Calculate firing rate from voltage.
        """
        # rectify
        rs = vs.copy()
        rs[rs <= 0] = 0
        return rs
    
    def sigmoid(self, x):
        """Sigmoid function using stored parameters."""
        return phi(self.steepness * (x - self.threshold))
    
    def reshape_to_matrix(self, vector):
        """Reshape vector to matrix corresponding to original lattice layout."""
        return np.reshape(vector, self.shape)