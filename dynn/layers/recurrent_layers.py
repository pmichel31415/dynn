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
from .. import activations
from .base_layers import ParametrizedLayer


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
        activation=activations.tanh,
        dropout=0.0
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
        self.Whx_p = self.pc.add_parameters(
            (self.hidden_dim, self.input_dim),
            name='Whx',
            init=NormalInit(scale_whx)
        )
        # Recurrent linear transform
        scale_whh = np.sqrt(1.0 / self.hidden_dim)
        self.Whh_p = self.pc.add_parameters(
            (self.hidden_dim, self.hidden_dim),
            name='Whh',
            init=NormalInit(scale_whh)
        )
        # Bias
        self.b_p = self.pc.add_parameters(
            (self.hidden_dim,), name='b', init=ZeroInit()
        )

    def init(self, test=False, update=True):
        """Initialize the layer before performing computation

        Args:
            test (bool, optional): If test mode is set to ``True``,
                dropout is not applied (default: ``True``)
            update (bool, optional): Whether to update the parameters
                (default: ``True``)
        """
        # Load weights in computation graph
        self.Whx = self.Whx_p.expr()
        self.Whh = self.Whh_p.expr()
        self.b = self.b_p.expr()
        # Initialize dropout mask (for recurrent dropout)
        self.test = test
        if not test and self.dropout > 0:
            self.dropout_mask_x = dy.dropout(
                dy.ones(self.input_dim), self.dropout)
            self.dropout_mask_h = dy.dropout(
                dy.ones(self.hidden_dim), self.dropout)

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
        self.Whx_p = self.pc.add_parameters(
            (self.hidden_dim * 4, self.input_dim),
            name="Whx",
            init=NormalInit(scale_whx)
        )
        # Output to hidden
        scale_whh = np.sqrt(2.0 / (5 * self.hidden_dim))
        self.Whh_p = self.pc.add_parameters(
            (self.hidden_dim * 4, self.hidden_dim),
            name="Whh",
            init=NormalInit(scale_whh)
        )
        # Bias
        self.b_p = self.pc.add_parameters(
            (self.hidden_dim * 4,), name="b", init=ZeroInit()
        )

    def init(self, test=False, update=True):
        """Initialize the layer before performing computation

        Args:
            test (bool, optional): If test mode is set to ``True``,
                dropout is not applied (default: ``True``)
            update (bool, optional): Whether to update the parameters
                (default: ``True``)
        """
        # Load weights in computation graph
        self.Whx = self.Whx_p.expr(update)
        self.Whh = self.Whh_p.expr(update)
        self.b = self.b_p.expr(update)
        # Initialize dropout mask
        self.test = test
        if not test and (self.dropout_x > 0 or self.dropout_h > 0):
            self.dropout_mask_x = dy.dropout(
                dy.ones(self.input_dim), self.dropout_x
            )
            self.dropout_mask_h = dy.dropout(
                dy.ones(self.hidden_dim), self.dropout_h
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
            gates = dy.vanilla_lstm_gates_dropout(
                x,
                h,
                self.Whx,
                self.Whh,
                self.b,
                self.dropout_mask_x,
                self.dropout_mask_h
            )
        else:
            gates = dy.vanilla_lstm_gates(x, h, self.Whx, self.Whh, self.b)
        new_c = dy.vanilla_lstm_c(c, gates)
        new_h = dy.vanilla_lstm_h(new_c, gates)
        return [new_h, new_c]

    def initial_value(self, batch_size=1):
        """Return two vectors of dimension `hidden_dim` filled with zeros

        Returns:
            tuple: two zero vectors for :math:`h_0` and :math:`c_0`
        """
        zero_vector = dy.zeros(self.hidden_dim, batch_size=batch_size)
        return zero_vector, zero_vector
