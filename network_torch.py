# the network.py file from Robert Yang's paper converted to pytorch by claude ai
"""Definition of the network model and various RNN cells"""

import os
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


def is_weight(name):
    """Check if a variable name represents a connection weight."""
    return 'weight' in name


def popvec(y):
    """Population vector read out.

    Assuming the last dimension is the dimension to be collapsed.

    Args:
        y: population output on a ring network. Numpy array (Batch, Units)

    Returns:
        Readout locations: Numpy array (Batch,)
    """
    pref = np.arange(0, 2*np.pi, 2*np.pi/y.shape[-1])  # preferences
    temp_sum = y.sum(axis=-1)
    temp_cos = np.sum(y*np.cos(pref), axis=-1)/temp_sum
    temp_sin = np.sum(y*np.sin(pref), axis=-1)/temp_sum
    loc = np.arctan2(temp_sin, temp_cos)
    return np.mod(loc, 2*np.pi)


def torch_popvec(y):
    """Population vector read-out in PyTorch."""
    num_units = y.size(-1)
    pref = torch.arange(0, 2*np.pi, 2*np.pi/num_units, dtype=torch.float32, device=y.device)  # preferences
    cos_pref = torch.cos(pref)
    sin_pref = torch.sin(pref)
    temp_sum = torch.sum(y, dim=-1, keepdim=True)
    temp_cos = torch.sum(y * cos_pref, dim=-1, keepdim=True) / temp_sum
    temp_sin = torch.sum(y * sin_pref, dim=-1, keepdim=True) / temp_sum
    loc = torch.atan2(temp_sin, temp_cos)
    return torch.fmod(loc, 2*np.pi)


def get_perf(y_hat, y_loc):
    """Get performance.

    Args:
      y_hat: Actual output. Numpy array (Time, Batch, Unit)
      y_loc: Target output location (-1 for fixation).
        Numpy array (Time, Batch)

    Returns:
      perf: Numpy array (Batch,)
    """
    if len(y_hat.shape) != 3:
        raise ValueError('y_hat must have shape (Time, Batch, Unit)')
    # Only look at last time points
    y_loc = y_loc[-1]
    y_hat = y_hat[-1]

    # Fixation and location of y_hat
    y_hat_fix = y_hat[..., 0]
    y_hat_loc = popvec(y_hat[..., 1:])

    # Fixating? Correctly saccading?
    fixating = y_hat_fix > 0.5

    original_dist = y_loc - y_hat_loc
    dist = np.minimum(abs(original_dist), 2*np.pi-abs(original_dist))
    corr_loc = dist < 0.2*np.pi

    # Should fixate?
    should_fix = y_loc < 0

    # performance
    perf = should_fix * fixating + (1-should_fix) * corr_loc * (1-fixating)
    return perf


class LeakyRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, alpha, sigma_rec=0, activation='softplus', w_rec_init='diag', rng=None):
        super(LeakyRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.alpha = alpha
        self.sigma = np.sqrt(2 / alpha) * sigma_rec
        self.activation = activation
        self.w_rec_init = w_rec_init

        if activation == 'softplus':
            self.act_fn = F.softplus
            self._w_in_start = 1.0
            self._w_rec_start = 0.5
        elif activation == 'tanh':
            self.act_fn = torch.tanh
            self._w_in_start = 1.0
            self._w_rec_start = 1.0
        elif activation == 'relu':
            self.act_fn = F.relu
            self._w_in_start = 1.0
            self._w_rec_start = 0.5
        elif activation == 'power':
            self.act_fn = lambda x: torch.pow(F.relu(x), 2)
            self._w_in_start = 1.0
            self._w_rec_start = 0.01
        elif activation == 'retanh':
            self.act_fn = lambda x: torch.tanh(F.relu(x))
            self._w_in_start = 1.0
            self._w_rec_start = 0.5
        else:
            raise ValueError('Unknown activation')

        if rng is None:
            rng = np.random.RandomState()
        self.rng = rng

        w_in0 = self.rng.randn(input_size, hidden_size) / np.sqrt(input_size) * self._w_in_start

        if w_rec_init == 'diag':
            w_rec0 = self._w_rec_start*np.eye(hidden_size)
        elif w_rec_init == 'randortho':
            w_rec0 = self._w_rec_start*tools.gen_ortho_matrix(hidden_size, rng=self.rng)
        elif w_rec_init == 'randgauss':
            w_rec0 = self._w_rec_start * self.rng.randn(hidden_size, hidden_size) / np.sqrt(hidden_size)

        matrix0 = np.concatenate((w_in0, w_rec0), axis=0)

        self.weight = Parameter(torch.Tensor(matrix0.T))
        self.bias = Parameter(torch.zeros(hidden_size))

    @property
    def state_size(self):
        return self.hidden_size

    @property
    def output_size(self):
        return self.hidden_size


    def forward(self, inputs, state):
        """Most basic RNN: output = new_state = act(W * input + U * state + B)."""
        state = state.squeeze(0)
        inputs = inputs.squeeze(0)

        if inputs.ndim == 2:
            mult_out = torch.matmul(torch.cat((inputs, state), dim=1), self.weight)
        else:
            mult_out = torch.matmul(torch.cat((inputs, state), dim=0), self.weight)

        gate_inputs = mult_out + self.bias

        noise = torch.randn_like(state) * self.sigma
        gate_inputs = gate_inputs + noise

        output = self.act_fn(gate_inputs)

        output = (1 - self.alpha) * state + self.alpha * output
        return output, output


class LeakyGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, alpha, sigma_rec=0, activation=None):
        super(LeakyGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.alpha = alpha
        self.sigma = np.sqrt(2 / alpha) * sigma_rec

        if activation is None:
            activation = torch.tanh
        self.activation = activation

        self.gate_weights = nn.Linear(input_size + hidden_size, 2 * hidden_size)
        self.candidate_weights = nn.Linear(input_size + hidden_size, hidden_size)
    
    @property
    def state_size(self):
        return self.hidden_size

    @property
    def output_size(self):
        return self.hidden_size


    def forward(self, inputs, state):
        state = state.squeeze(0)
        inputs = inputs.squeeze(0)

        if inputs.ndim == 2:
            gate_inputs = torch.cat((inputs, state), dim=1)
        else:
            gate_inputs = torch.cat((inputs, state), dim=0)

        gates = self.gate_weights(gate_inputs)
        gates = torch.sigmoid(gates)
        r, u = gates.chunk(2, 1)

        r_state = r * state

        candidate = self.candidate_weights(torch.cat((inputs, r_state), dim=1))
        candidate = self.activation(candidate)

        noise = torch.randn_like(state) * self.sigma
        candidate = candidate + noise

        new_h = (1 - self.alpha * u) * state + self.alpha * u * candidate
        return new_h, new_h


class LeakyRNNCellSeparateInput(nn.Module):
    def __init__(self, hidden_size, alpha, sigma_rec=0, activation='softplus', w_rec_init='diag', rng=None):
        super(LeakyRNNCellSeparateInput, self).__init__()
        self.hidden_size = hidden_size
        self.alpha = alpha
        self.sigma = np.sqrt(2 / alpha) * sigma_rec
        self.activation = activation
        self.w_rec_init = w_rec_init

        if activation == 'softplus':
            self.act_fn = F.softplus
            self._w_rec_start = 0.5
        elif activation == 'relu':
            self.act_fn = F.relu
            self._w_rec_start = 0.5
        else:
            raise ValueError('Unknown activation')

        if rng is None:
            rng = np.random.RandomState()
        self.rng = rng

        if w_rec_init == 'diag':
            w_rec0 = self._w_rec_start*np.eye(hidden_size)
        elif w_rec_init == 'randortho':
            w_rec0 = self._w_rec_start*tools.gen_ortho_matrix(hidden_size, rng=self.rng)
        elif w_rec_init == 'randgauss':
            w_rec0 = self._w_rec_start * self.rng.randn(hidden_size, hidden_size) / np.sqrt(hidden_size)
        else:
            raise ValueError

        self.weight = Parameter(torch.Tensor(w_rec0.T))
        self.bias = Parameter(torch.zeros(hidden_size))

    @property
    def state_size(self):
        return self.hidden_size

    @property
    def output_size(self):
        return self.hidden_size


    def forward(self, inputs, state):
        """output = new_state = act(input + U * state + B)."""
        state = state.squeeze(0)
        inputs = inputs.squeeze(0)

        if inputs.ndim == 2:
            mult_out = torch.matmul(state, self.weight)
        else:
            mult_out = torch.matmul(state, self.weight)

        gate_inputs = mult_out + inputs + self.bias

        noise = torch.randn_like(state) * self.sigma
        gate_inputs = gate_inputs + noise

        output = self.act_fn(gate_inputs)

        output = (1 - self.alpha) * state + self.alpha * output
        return output, output


class Model(nn.Module):
    def __init__(self, model_dir, hp=None, sigma_rec=None, dt=None):
        super(Model, self).__init__()

        if hp is None:
            hp = tools.load_hp(model_dir)
            if hp is None:
                raise ValueError('No hp found for model_dir {:s}'.format(model_dir))

        torch.manual_seed(hp['seed'])
        self.rng = np.random.RandomState(hp['seed'])

        if sigma_rec is not None:
            print('Overwrite sigma_rec with {:0.3f}'.format(sigma_rec))
            hp['sigma_rec'] = sigma_rec

        if dt is not None:
            print('Overwrite original dt with {:0.1f}'.format(dt))
            hp['dt'] = dt

        hp['alpha'] = 1.0*hp['dt']/hp['tau']

        if hp['in_type'] != 'normal':
            raise ValueError('Only support in_type ' + hp['in_type'])

        self._build(hp)

        self.model_dir = model_dir
        self.hp = hp

    def _build(self, hp):
        if 'use_separate_input' in hp and hp['use_separate_input']:
            self._build_separate(hp)
        else:
            self._build_fused(hp)

        self.weight_list = [v for k, v in self.named_parameters() if is_weight(k)]

        if 'use_separate_input' in hp and hp['use_separate_input']:
            self._set_weights_separate(hp)
        else:
            self._set_weights_fused(hp)

        # Regularization terms
        self.cost_reg = torch.tensor(0.)
        if hp['l1_h'] > 0:
            self.cost_reg += torch.mean(torch.abs(self.h)) * hp['l1_h']
        if hp['l2_h'] > 0:
            self.cost_reg += torch.sum(torch.pow(self.h, 2)) * hp['l2_h']

        if hp['l1_weight'] > 0:
            self.cost_reg += hp['l1_weight'] * torch.sum(torch.tensor([torch.mean(torch.abs(v)) for v in self.weight_list]))
        if hp['l2_weight'] > 0:
            self.cost_reg += hp['l2_weight'] * torch.sum(torch.tensor([torch.sum(torch.pow(v, 2)) for v in self.weight_list]))

        # Create an optimizer.
        if 'optimizer' not in hp or hp['optimizer'] == 'adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=hp['learning_rate'])
        elif hp['optimizer'] == 'sgd':
            self.optimizer = torch.optim.SGD(self.parameters(), lr=hp['learning_rate'])

    def _build_fused(self, hp):
        n_input = hp['n_input']
        n_rnn = hp['n_rnn']
        n_output = hp['n_output']

        if hp['activation'] == 'power':
            f_act = lambda x: torch.pow(F.relu(x), 2)
        elif hp['activation'] == 'retanh':
            f_act = lambda x: torch.tanh(F.relu(x))
        elif hp['activation'] == 'relu+':
            f_act = lambda x: F.relu(x + 1.0)
        else:
            f_act = getattr(F, hp['activation'])

        # Recurrent activity
        if hp['rnn_type'] == 'LeakyRNN':
            self.rnn_cell = LeakyRNNCell(n_input, n_rnn, hp['alpha'], sigma_rec=hp['sigma_rec'],
                                         activation=hp['activation'], w_rec_init=hp['w_rec_init'], rng=self.rng)
        elif hp['rnn_type'] == 'LeakyGRU':
            self.rnn_cell = LeakyGRUCell(n_input, n_rnn, hp['alpha'], sigma_rec=hp['sigma_rec'], activation=f_act)
        elif hp['rnn_type'] == 'LSTM':
            self.rnn_cell = nn.LSTMCell(n_input, n_rnn)
            self.rnn_cell.activation = f_act
        elif hp['rnn_type'] == 'GRU':
            self.rnn_cell = nn.GRUCell(n_input, n_rnn)
            self.rnn_cell.activation = f_act
        else:
            raise NotImplementedError("rnn_type must be one of LeakyRNN, LeakyGRU, LSTM, GRU")

        self.output_layer = nn.Linear(n_rnn, n_output)

    def _set_weights_fused(self, hp):
        """Set model attributes for several weight variables."""
        n_input = hp['n_input']
        n_rnn = hp['n_rnn']
        n_output = hp['n_output']

        for name, param in self.named_parameters():
            if 'rnn' in name:
                if 'weight' in name:
                    self.w_rec = param[n_input:, :]
                    self.w_in = param[:n_input, :]
                else:
                    self.b_rec = param
            else:
                assert 'output' in name
                if 'weight' in name:
                    self.w_out = param
                else:
                    self.b_out = param

        # check if the recurrent and output connection has the correct shape
        if self.w_out.shape != (n_rnn, n_output):
            raise ValueError('Shape of w_out should be ' +
                                str((n_rnn, n_output)) + ', but found ' +
                                str(self.w_out.shape))
        if self.w_rec.shape != (n_rnn, n_rnn):
            raise ValueError('Shape of w_rec should be ' +
                                str((n_rnn, n_rnn)) + ', but found ' +
                                str(self.w_rec.shape))
        if self.w_in.shape != (n_input, n_rnn):
            raise ValueError('Shape of w_in should be ' +
                                str((n_input, n_rnn)) + ', but found ' +
                                str(self.w_in.shape))

    def _build_separate(self, hp):
        n_input = hp['n_input']
        n_rnn = hp['n_rnn']
        n_output = hp['n_output']

        sensory_size = hp['rule_start']
        rule_size = hp['n_rule']

        self.sensory_rnn_input = nn.Linear(sensory_size, n_rnn)

        if 'mix_rule' in hp and hp['mix_rule'] is True:
            # rotate rule matrix
            kernel_initializer = torch.nn.init.orthogonal_
            self.mix_rule = nn.Linear(rule_size, rule_size, bias=False)
            self.mix_rule.weight.requires_grad = False
            kernel_initializer(self.mix_rule.weight)

        self.rule_rnn_input = nn.Linear(rule_size, n_rnn, bias=False)

        # Recurrent activity
        self.rnn_cell = LeakyRNNCellSeparateInput(n_rnn, hp['alpha'], sigma_rec=hp['sigma_rec'],
                                                    activation=hp['activation'], w_rec_init=hp['w_rec_init'], rng=self.rng)

        self.output_layer = nn.Linear(n_rnn, n_output)

    def _set_weights_separate(self, hp):
        """Set model attributes for several weight variables."""
        n_rnn = hp['n_rnn']
        n_output = hp['n_output']

        for name, param in self.named_parameters():
            if 'rnn_cell' in name:
                if 'weight' in name:
                    self.w_rec = param
                else:
                    self.b_rec = param
            elif 'sensory_rnn_input' in name:
                if 'weight' in name:
                    self.w_sen_in = param
                else:
                    self.b_in = param
            elif 'rule_rnn_input' in name:
                self.w_rule = param
            else:
                assert 'output' in name
                if 'weight' in name:
                    self.w_out = param
                else:
                    self.b_out = param

        # check if the recurrent and output connection has the correct shape
        if self.w_out.shape != (n_rnn, n_output):
            raise ValueError('Shape of w_out should be ' +
                                str((n_rnn, n_output)) + ', but found ' +
                                str(self.w_out.shape))
        if self.w_rec.shape != (n_rnn, n_rnn):
            raise ValueError('Shape of w_rec should be ' +
                                str((n_rnn, n_rnn)) + ', but found ' +
                                str(self.w_rec.shape))
        if self.w_sen_in.shape != (hp['rule_start'], n_rnn):
            raise ValueError('Shape of w_sen_in should be ' +
                                str((hp['rule_start'], n_rnn)) + ', but found ' +
                                str(self.w_sen_in.shape))
        if self.w_rule.shape != (hp['n_rule'], n_rnn):
            raise ValueError('Shape of w_in should be ' +
                                str((hp['n_rule'], n_rnn)) + ', but found ' +
                                str(self.w_rule.shape))

    def forward(self, x, y=None, c_mask=None):
        n_batch, n_time, _ = x.shape

        self.h = torch.zeros(n_time + 1, n_batch, self.hp['n_rnn'], device=x.device)

        if 'use_separate_input' in self.hp and self.hp['use_separate_input']:
            sensory_inputs, rule_inputs = torch.split(x, [self.hp['rule_start'], self.hp['n_rule']], dim=-1)

            sensory_rnn_inputs = self.sensory_rnn_input(sensory_inputs)

            if 'mix_rule' in self.hp and self.hp['mix_rule'] is True:
                rule_inputs = self.mix_rule(rule_inputs)

            rule_rnn_inputs = self.rule_rnn_input(rule_inputs)

            rnn_inputs = sensory_rnn_inputs + rule_rnn_inputs
        else:
            rnn_inputs = x

        # Run RNN
        for t in range(n_time):
            self.h[t + 1], _ = self.rnn_cell(rnn_inputs[t], self.h[t])

        h_shaped = self.h[1:].reshape(-1, self.hp['n_rnn'])

        y_hat = self.output_layer(h_shaped)

        if self.hp['loss_type'] == 'lsq':
            # Least-square loss
            y_hat = torch.sigmoid(y_hat)
            y_shaped = y.view(-1, self.hp['n_output'])
            self.cost_lsq = torch.mean(
                torch.pow((y_shaped - y_hat) * c_mask.view(-1, self.hp['n_output']), 2))
        else:
            y_hat = torch.softmax(y_hat, dim=-1)
            y_shaped = y.view(-1, self.hp['n_output'])
            # Cross-entropy loss
            self.cost_lsq = torch.mean(
                -c_mask.view(-1, 1) * torch.sum(y_shaped * torch.log(y_hat), dim=-1))

        self.y_hat = y_hat.view(-1, n_batch, self.hp['n_output'])

        y_hat_fix, y_hat_ring = torch.split(self.y_hat, [1, self.hp['n_output'] - 1], dim=-1)
        self.y_hat_loc = torch_popvec(y_hat_ring)

        return self.y_hat, self.h[1:]

    def set_optimizer(self, extra_cost=None, var_list=None):
        """Recompute the optimizer to reflect the latest cost function.

        This is useful when the cost function is modified throughout training

        Args:
            extra_cost : torch.Tensor,
            added to the lsq and regularization cost
        """
        cost = self.cost_lsq + self.cost_reg
        if extra_cost is not None:
            cost += extra_cost

        if var_list is None:
            var_list = list(self.parameters())

        print('Variables being optimized:')
        for v in var_list:
            print(v.size())

        self.optimizer = torch.optim.Adam(var_list, lr=self.hp['learning_rate'])

    def lesion_units(self, units, verbose=False):
        """Lesion units given by units

        Args:
            units : can be None, an integer index, or a list of integer indices
        """

        # Convert to numpy array
        if units is None:
            return
        elif not hasattr(units, '__iter__'):
            units = np.array([units])
        else:
            units = np.array(units)

        # This lesioning will work for both RNN and GRU
        n_input = self.hp['n_input']
        for name, param in self.named_parameters():
            if 'weight' in name:
                # Connection weights
                if 'output' in name:
                    # output weights
                    param.data[units, :] = 0
                elif 'rnn' in name:
                    # recurrent weights
                    param.data[n_input + units, :] = 0

        if verbose:
            print('Lesioned units:')
            print(units)