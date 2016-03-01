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
        
        super(self.__class__, self).__init__()
    
    def step(self, scalar_drive=0, matrix_drive=None):
        """
        Step forward one instant in time, optionally providing drive to the network.
        
        :param scalar_drive: same drive to whole network
        :param matrix_drive: different drives to each neuron
        """
        
        if matrix_drive is None:
            drive = scalar_drive
        else:
            drive = matrix_drive.flatten() + scalar_drive
        
        node_inputs = self.w.dot(self.rs) + drive
        
        # compute new voltages
        new_vs = self.vs.copy()
        
        # decrement all activations greater than 1
        new_vs[self.vs > 1] -= 1
        # set all smallest activations to self.inactivation_strength
        new_vs[self.vs == 1] = self.inactivation_strength
        # increment all negative activations
        new_vs[self.vs < 0] += 1
        
        # probabilistically activate nodes receiving highest input
        self.activation_probabilities = self.sigmoid(node_inputs)
        # only allow nodes with zero activation value to have the chance to activate
        self.activation_probabilities[self.vs != 0] = 0
        # sample which nodes have become active, and set their voltage to self.activation_strength
        new_vs[np.random.rand(self.n_nodes) < self.activation_probabilities] = self.activation_strength
        
        # store voltages
        self.vs = new_vs
        
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
    
    @property
    def vs_matrix(self):
        return self.vs.reshape(self.shape)
    
    @vs_matrix.setter
    def vs_matrix(self, vs_matrix):
        self.vs = vs_matrix.flatten()
        
    @property
    def rs_matrix(self):
        return self.rs.reshape(self.shape)
    

class NeuralSandPileModel1(VoltageFiringRateModel):
    """Class representing a "neural" version of Bak, Tang, and Wiesenfeld's 1987 sandpile model,
    which has the key property of self-organized criticality."""
    
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
    
    def __init__(self, shape, threshold, weight_type, **weight_matrix_kwargs):
        
        self.shape = shape
        self.threshold = threshold
        
        self.n_nodes = np.prod(shape)
        
        self.w = .125 * self.make_weight_matrix(shape, weight_type, **weight_matrix_kwargs)
        
        super(self.__class__, self).__init__()
    
    def step(self, scalar_drive=0, matrix_drive=None):
        """
        Step forward one instant in time, optionally providing drive to the network.
        
        :param scalar_drive: same drive to whole network
        :param matrix_drive: different drives to each neuron
        """
        
        if matrix_drive is None:
            drive = scalar_drive
        else:
            drive = matrix_drive.flatten() + scalar_drive
            
        node_inputs = self.w.dot(self.rs) + drive
        node_inputs[self.rs > 0] = -self.rs[self.rs > 0]
        self.vs += node_inputs
        
        self.record_data()
    
    def randomize_voltages(self, v_max, v_mean):
        self.vs = np.random.binomial(int(v_max), p=v_mean/int(v_max), size=(self.n_nodes,))
        
    def rate_from_voltage(self, vs):
        """
        Calculate firing rate from voltage.
        """
        rs = vs.copy()
        rs[rs < self.threshold] = 0
        return rs
    
    @property
    def vs_matrix(self):
        return self.vs.reshape(self.shape)
    
    @vs_matrix.setter
    def vs_matrix(self, vs_matrix):
        self.vs = vs_matrix.flatten()
        
    @property
    def rs_matrix(self):
        return self.rs.reshape(self.shape)
    

class RecurrentSoftMaxModel(object):
    """
    Network in which one node fires at a time, chosen through soft-max function.
    The "inequality" of firing probabilities given inputs to all the nodes is determined by the gain.
    If the gain is high, nodes with higher inputs are more preferred.
    
    :param weights: weight matrix
    :param gain: gain going into softmax function
    """
    
    def __init__(self, weights, gain, shape=None):
        self.w = weights
        self.n_nodes = weights.shape[0]
        self.gain = gain
        self.shape = shape

        self.vs = None
        self.rs = None

        self.vs_history = []
        
        self.rs_history = []  # since firing rate is no longer a deterministic function of voltages
        
    def record_data(self):
        if self.store_voltages:
            self.vs_history.append(self.vs)
            self.rs_history.append(self.rs)
            
    def step(self, drive=0):
        """
        Step forward one time step, optionally providing drive to the network.
        
        :param drive: network drive (can be scalar or 1D array)
        """
        if self.rs is None:
            rs = np.zeros((self.n_nodes,), dtype=float)
        else:
            rs = self.rs

        inputs = self.w.dot(rs) + drive
        self.vs = self.gain * inputs
        self.rs = self.rate_from_voltage(self.vs)
        
        self.record_data()

    def get_active_idx(self, vs):
        """
        Randomly sample which node should be active given all the nodes' voltages.
        """
        
        p_fire = np.exp(vs)
        p_fire /= p_fire.sum()
        active_idx = np.random.choice(range(self.n_nodes), p=p_fire)

        return active_idx
    
    def rate_from_voltage(self, vs):
        """
        Calculate firing rate from voltage.
        """
        
        rs = np.zeros((self.n_nodes,), dtype=float)
        rs[self.get_active_idx(vs)] = 1.
        
        return rs
    
    @property
    def vs_matrix(self):
        return self.vs.reshape(self.shape)
    
    @vs_matrix.setter
    def vs_matrix(self, vs_matrix):
        self.vs = vs_matrix.flatten()
    
    @property
    def rs_matrix(self):
        return self.rs.reshape(self.shape)
    
    
class RecurrentSoftMaxLingeringModel(RecurrentSoftMaxModel):
    """
    Network similar to RecurrentSoftMax model, except that node activity lingers after activation.
    
    :param weights: weight matrix
    :param gain: gain going into softmax function
    :param lingering_input_value: lingering input value
    :param lingering_timescale: timescale of lingering input
    """
    
    def __init__(self, weights, gain, lingering_input_value, lingering_timescale, shape=None):
        
        super(self.__class__, self).__init__(weights, gain, shape)
        
        self.lingering_input_value = lingering_input_value
        self.lingering_timescale = lingering_timescale

        self.lingering_inputs = np.zeros((self.n_nodes,), dtype=float)
        self.lingering_inputs_counter = np.zeros((self.n_nodes,), dtype=float)
        
    def step(self, drive=0):
        """
        Step forward one time step, optionally providing drive to the network.
        
        :param drive: network drive (can be scalar or 1D array)
        """
        if self.rs is None:
            rs = np.zeros((self.n_nodes,), dtype=float)
        else:
            rs = self.rs

        inputs = self.w.dot(rs) + self.lingering_inputs + drive

        # decrease lingering counter and set lingering inputs to zero if necessary
        self.lingering_inputs_counter[self.lingering_inputs_counter > 0] -= 1
        self.lingering_inputs[self.lingering_inputs_counter == 0] = 0

        self.vs = self.gain * inputs
        self.rs = self.rate_from_voltage(self.vs)
        
        self.record_data()
        
    def rate_from_voltage(self, vs):
        """
        Calculate firing rate from voltage.
        """
        
        active_idx = self.get_active_idx(vs)
        
        rs = np.zeros((self.n_nodes,), dtype=float)
        rs[active_idx] = 1.
        
        self.lingering_inputs[active_idx] = self.lingering_input_value
        self.lingering_inputs_counter[active_idx] = self.lingering_timescale
        
        return rs


class RecurrentSoftMaxLingeringSTDPModel(RecurrentSoftMaxModel):
    """
    Network similar to recurrent soft max lingering model, except that lingering has finite time scale
    and the network has STDP-like synaptic plasticity.

    Here, the weight to node i from node j is increased by an amount alpha * (w_max - w_ij) if
    node i activates immediately after node j.

    :param weights: weight matrix
    :param gain: gain going into softmax function
    :param lingering_input_value: lingering input value
    :param lingering_timescale: timescale of lingering input
    :param w_max: maximum weight
    :param alpha: learning rate
    :param shape: shape of network if it is layed out on a grid
    """

    def __init__(self, weights, gain, lingering_input_value, lingering_timescale, w_max, alpha, shape=None):

        super(self.__class__, self).__init__(weights, gain, shape)

        self.lingering_input_value = lingering_input_value
        self.lingering_timescale = lingering_timescale
        self.w_max = w_max
        self.alpha = alpha
        self.lingering_inputs = np.zeros((self.n_nodes,), dtype=float)
        self.lingering_inputs_counter = np.zeros((self.n_nodes,), dtype=float)

    def step(self, drive=0):
        """
        Step forward one time step, optionally providing drive to the network.

        :param drive: network drive (can be scalar or 1D array)
        """
        if self.rs is None:
            rs = np.zeros((self.n_nodes,), dtype=float)
        else:
            rs = self.rs

        # calculate inputs
        inputs = self.w.dot(rs) + self.lingering_inputs + drive

        # decrease lingering counter and set lingering inputs to zero if necessary
        self.lingering_inputs_counter[self.lingering_inputs_counter > 0] -= 1
        self.lingering_inputs[self.lingering_inputs_counter == 0] = 0

        # store previous firing rates for STDP purposes
        rs_prev = rs.copy()

        # calculate new voltages and firing rates
        self.vs = self.gain * inputs
        self.rs = self.rate_from_voltage(self.vs)

        # run STDP (only operable for one connection per timestep)
        if rs_prev.sum() > 0:
            src = rs_prev.nonzero()[0][0]
            targ = self.rs.nonzero()[0][0]
            if self.w[targ, src] > 0:
                self.w[targ, src] += self.alpha * (self.w_max - self.w[targ, src])

        self.record_data()

    def rate_from_voltage(self, vs):
        """
        Calculate firing rate from voltage.
        """

        active_idx = self.get_active_idx(vs)

        rs = np.zeros((self.n_nodes,), dtype=float)
        rs[active_idx] = 1.

        self.lingering_inputs[active_idx] = self.lingering_input_value
        self.lingering_inputs_counter[active_idx] = self.lingering_timescale

        return rs