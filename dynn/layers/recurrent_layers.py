#!/usr/bin/env python3
"""
Recurrent layers
================

The particularity of recurrent is that their output can be fed back as input.
This includes common recurrent cells like the Elman RNN or the LSTM.
"""

import numpy as np
import dynet as dy

from ..parameter_initialization import ZeroInit, NormalInit
from ..activations import tanh, sigmoid
from .base_layers import BaseLayer, ParametrizedLayer


class RecurrentCell(object):
    """Base recurrent cell interface

    Recurrent cells must provide a default initial value for their recurrent
    state (eg. all zeros)
    """

    def __init__(self, *args, **kwargs):
        pass

    def initial_value(self, batch_size=1):
        """Initial value of the recurrent state. Should return a list."""
        raise NotImplementedError()

    def get_output(self, state):
        """Get the cell's output from the list of states.

        For example this would return ``h`` from ``h,c`` in the case of the
        LSTM"""
        raise NotImplementedError()


class StackedRecurrentCells(BaseLayer, RecurrentCell):
    """This implements a stack of recurrent layers

    The recurrent state of the resulting cell is the list of the states
    of all the sub-cells. For example for a stack of 2 LSTM cells the
    resulting state will be ``[h_1, c_1, h_2, c_2]``

    Example:

    .. code-block:: python

        # Parameter collection
        pc = dy.ParameterCollection()
        # Stacked recurrent cell
        stacked_cell = StackedRecurrentCells(
            LSTM(pc, 10, 15),
            LSTM(pc, 15, 5),
            ElmanRNN(pc, 5, 20),
        )
        # Inputs
        dy.renew_cg()
        x = dy.random_uniform(10, batch_size=5)
        # Initialize layer
        stacked_cell.init(test=False)
        # Initial state: [h_1, c_1, h_2, c_2, h_3] of sizes [15, 15, 5, 5, 20]
        init_state = stacked_cell.initial_value()
        # Run the cell on the input.
        new_state = stacked_cell(x, *init_state)
        # Get the final output (h_3 of size 20)
        h = stacked_cell.get_output(new_state)
        """

    def __init__(self, *cells):
        super(StackedRecurrentCells, self).__init__("stacked-recurrent")
        for cell_idx, cell in enumerate(cells):
            if not isinstance(cell, RecurrentCell):
                raise ValueError(
                    f"Expected RecurrentCell, got {cell.__class__} in "
                    f"StackedRecurrentCells constructor (cell #{cell_idx}))."
                )
            setattr(self, f"cell_{cell_idx}", cell)
        self.cells = cells
        self.state_sizes = [len(cell.initial_value()) for cell in self.cells]

        # This is used to get the states for a specific layer
        cum_state_size = np.cumsum([0] + self.state_sizes)
        self.state_slices = [
            slice(start, stop, None)
            for start, stop in zip(cum_state_size[:-1], cum_state_size[1:])
        ]

    def initial_value(self, batch_size=1):
        """Initial value of the recurrent state."""
        return [state for cell in self.cells
                for state in cell.initial_value(batch_size)]

    def get_output(self, state):
        """Get the output of the last cell"""
        last_cell_state = state[self.state_slices[-1]]
        return self.cells[-1].get_output(last_cell_state)

    def __call__(self, x, *state):
        """Compute the cell's output from the list of states and an input expression

        Args:
            x (:py:class:`dynet.Expression`): Input vector

        Returns:
            list: new recurrent state
        """

        # New state
        new_state = []
        # Iterate over the cells
        for n_cell, cell in enumerate(self.cells):
            # Retrieve the previous state for this layer
            cell_state = state[self.state_slices[n_cell]]
            # Run the cell and get the new state
            new_cell_state = cell(x, *cell_state)
            # Retrieve the output value to feedback as input to the next layer
            x = cell.get_output(new_cell_state)
            # Add the cell state to the new state
            new_state.extend(new_cell_state)

        return new_state


class ElmanRNN(ParametrizedLayer, RecurrentCell):
    """The standard Elman RNN cell:

    :math:`h_{t}=\sigma(W_{hh}h_{t-1} + W_{hx}x_{t} + b)`

    Args:
        pc (:py:class:`dynet.ParameterCollection`): Parameter collection to
            hold the parameters
        input_dim (int): Input dimension
        output_dim (int): Output (hidden) dimension
        activation (function, optional): Activation function :math:`sigma`
            (default: :py:func:`dynn.activations.tanh`)
        dropout (float, optional):  Dropout rate (default 0)
    """

    def __init__(
        self,
        pc,
        input_dim,
        hidden_dim,
        activation=tanh,
        dropout=0.0,
        Whx=None,
        Whh=None,
        b=None,
    ):
        super(ElmanRNN, self).__init__(pc, "elman-rnn")
        # Hyper parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.activation = activation

        # Parameters
        # Input linear transform
        scale_whx = np.sqrt(2.0 / (self.hidden_dim + self.input_dim))
        self.add_parameters(
            "Whx",
            (self.hidden_dim, self.input_dim),
            init=NormalInit(mean=0, std=scale_whx),
            param=Whx,
        )
        # Recurrent linear transform
        scale_whh = np.sqrt(1.0 / self.hidden_dim)
        self.add_parameters(
            "Whh",
            (self.hidden_dim, self.hidden_dim),
            init=NormalInit(mean=0, std=scale_whh),
            param=Whh,
        )
        # Bias
        self.add_parameters("b", self.hidden_dim, init=ZeroInit(), param=b)

    def init_layer(self, test=True, update=False):
        super(ElmanRNN, self).init_layer(test=test, update=update)
        # Initialize dropout mask (for recurrent dropout)
        if not test and self.dropout > 0:
            self.dropout_mask_x = dy.dropout(
                dy.ones(self.input_dim),
                self.dropout
            )
            self.dropout_mask_h = dy.dropout(
                dy.ones(self.hidden_dim),
                self.dropout
            )

    def __call__(self, x, h):
        """Perform the recurrent update.

        Args:
            x (:py:class:`dynet.Expression`): Input vector
            h (:py:class:`dynet.Expression`): Previous recurrent vector

        Returns:
            :py:class:`dynet.Expression`: Next recurrent state
                :math:`h_{t}=\sigma(W_{hh}h_{t-1} + W_{hx}x_{t} + b)`
        """
        # Dropout
        if not self.test and self.dropout > 0:
            x = dy.cmult(x, self.dropout_mask_x)
            h = dy.cmult(h, self.dropout_mask_h)
        # Compute the new hidden state
        new_h = dy.affine_transform([self.b, self.Whh, h, self.Whx, x])
        return [self.activation(new_h)]

    def initial_value(self, batch_size=1):
        """Return a vector of dimension `hidden_dim` filled with zeros

        Returns:
            :py:class:`dynet.Expression`: Zero vector
        """
        return [dy.zeros(self.hidden_dim, batch_size=batch_size)]

    def get_output(self, state):
        return state[0]


class LSTM(ParametrizedLayer, RecurrentCell):
    """Standard LSTM

    Args:
        pc (:py:class:`dynet.ParameterCollection`): Parameter collection to
            hold the parameters
        input_dim (int): Input dimension
        output_dim (int): Output (hidden) dimension
        dropout_x (float, optional): Input dropout rate (default 0)
        dropout_h (float, optional): Recurrent dropout rate (default 0)
    """

    def __init__(
        self,
        pc,
        input_dim,
        hidden_dim,
        dropout_x=0.0,
        dropout_h=0.0,
        Whx=None,
        Whh=None,
        b=None,
    ):
        super(LSTM, self).__init__(pc, "compact-lstm")
        # Hyperparameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_x = dropout_x
        self.dropout_h = dropout_h

        # Parameters
        # Input to hidden
        scale_whx = np.sqrt(2.0 / (4 * self.hidden_dim + self.input_dim))
        self.add_parameters(
            "Whx",
            (self.hidden_dim * 4, self.input_dim),
            init=NormalInit(mean=0, std=scale_whx),
            param=Whx,
        )
        # Output to hidden
        scale_whh = np.sqrt(2.0 / (5 * self.hidden_dim))
        self.add_parameters(
            "Whh",
            (self.hidden_dim * 4, self.hidden_dim),
            init=NormalInit(mean=0, std=scale_whh),
            param=Whh,
        )
        # Bias
        self.add_parameters("b", self.hidden_dim * 4, init=ZeroInit(), param=b)

    def init_layer(self, test=True, update=False):
        super(LSTM, self).init_layer(test=test, update=update)
        # Initialize dropout mask (for recurrent dropout)
        if not test and (self.dropout_x > 0 or self.dropout_h > 0):
            self.dropout_mask_x = dy.dropout(
                dy.ones(self.input_dim),
                self.dropout_x,
            )
            self.dropout_mask_h = dy.dropout(
                dy.ones(self.hidden_dim),
                self.dropout_h,
            )

    def __call__(self, x, h, c):
        """Perform the recurrent update.

        Args:
            x (:py:class:`dynet.Expression`): Input vector
            h (:py:class:`dynet.Expression`): Previous recurrent vector
            c (:py:class:`dynet.Expression`): Previous cell state vector

        Returns:
            tuple::py:class:`dynet.Expression` for the ext recurrent states
                ``h`` and ``c``
        """
        if not self.test and (self.dropout_x > 0 or self.dropout_h > 0):
            x = dy.cmult(self.dropout_mask_x, x)
            h = dy.cmult(self.dropout_mask_h, h)
        gates = dy.affine_transform([self.b, self.Whx, x, self.Whh, h])
        # Split gates and apply nonlinearities
        i = sigmoid(gates[:self.hidden_dim])
        f = sigmoid(gates[self.hidden_dim:2*self.hidden_dim] + 1)
        o = sigmoid(gates[2*self.hidden_dim:3*self.hidden_dim])
        g = tanh(gates[3*self.hidden_dim:])
        # New cell state
        new_c = dy.cmult(f, c) + dy.cmult(i, g)
        new_h = dy.cmult(tanh(new_c), o)
        return [new_h, new_c]

    def initial_value(self, batch_size=1):
        """Return two vectors of dimension `hidden_dim` filled with zeros

        Returns:
            tuple: two zero vectors for :math:`h_0` and :math:`c_0`
        """
        zero_vector = dy.zeros(self.hidden_dim, batch_size=batch_size)
        return zero_vector, zero_vector

    def get_output(self, state):
        return state[0]


class StackedLSTM(StackedRecurrentCells):
    """Stacked LSTMs

    Args:
        pc (:py:class:`dynet.ParameterCollection`): Parameter collection to
            hold the parameters
        num_layers (int): Number of layers
        input_dim (int): Input dimension
        output_dim (int): Output (hidden) dimension
        dropout_x (float, optional): Input dropout rate (default 0)
        dropout_h (float, optional): Recurrent dropout rate (default 0)
    """

    def __init__(
        self,
        pc,
        num_layers,
        input_dim,
        hidden_dim,
        dropout_x=0.0,
        dropout_h=0.0,
    ):
        # Create recurrent cells
        dims = [input_dim] + [hidden_dim] * num_layers
        lstm_cells = [
            LSTM(pc, di, dh, dropout_x=dropout_x, dropout_h=dropout_h)
            for di, dh in zip(dims[:-1], dims[1:])
        ]
        # Initialize
        super(StackedLSTM, self).__init__(*lstm_cells)
