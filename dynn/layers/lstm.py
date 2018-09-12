#!/usr/bin/env python3
"""FIXME"""

import numpy as np
import dynet as dy

from ..parameter_initialization import ZeroInit, OneInit
from .base_layer import BaseLayer


class NaryTreeLSTMCell(BaseLayer):
    """N-ary TreeLSTM as introduced in Tai et al, 2015

    Args:
        arity (int): Number of hidden children not counting the input
            (eg. a standard LSTM is a unary TreeLSTM)
        input_dim (int): Input dimension
            (it is assumed that all inputs have the same dimension)
        hidden_dim (int): Hidden dimension (also the output dimension)
        pc (dynet.ParameterCollection): Parameter collection to
            hold the parameters
        dropout (float, optional):  Dropout rate (default 0)
        diagonal(bool, optional): If set to true, the forget gate for
            child i will only be computed based on this children's state
            at the previous step (vs all other children's previous states).
            This reduces the number of parameters [description]
    """

    def __init__(
        self,
        arity,
        input_dim,
        hidden_dim,
        pc,
        dropout=0.0,
        diagonal=False,
    ):
        super(NaryTreeLSTMCell, self).__init__(pc, "{arity}-treelstm")
        # Arity
        self.arity = arity
        assert self.arity >= 0, f"Arity {self.arity} not supported"
        # Output dimension
        self.dh = hidden_dim
        # Input dimension
        self.di = input_dim
        # Whether to use diagonal forget gates
        # (ie no connections between h_i and f_j if i!=j)
        self.diagonal = diagonal
        # Dimensions for the parameters
        self.incoming_dim = self.di + self.dh * self.arity
        self.diagonal_dim = (
            self.di + self.dh) if self.diagonal else self.incoming_dim
        # Dropout
        self.dropout = dropout
        # Parameters
        self.Wi_p = self.pc.add_parameters(
            (self.dh, self.incoming_dim),
            name="Wi"
        )
        self.Wo_p = self.pc.add_parameters(
            (self.dh, self.incoming_dim),
            name="Wo"
        )
        self.Wg_p = self.pc.add_parameters(
            (self.dh, self.incoming_dim),
            name="Wg"
        )
        self.Wf_p = [
            self.pc.add_parameters(
                (self.dh, self.diagonal_dim), name=f"Wf{i}"
            )
            for i in range(self.arity)
        ]
        # Biases
        self.bi_p = self.pc.add_parameters(
            self.dh, name="bi", init=ZeroInit)
        self.bo_p = self.pc.add_parameters(
            self.dh, name="bo", init=ZeroInit)
        self.bg_p = self.pc.add_parameters(
            self.dh, name="bg", init=ZeroInit)
        self.bf_p = [
            self.pc.add_parameters(
                self.dh, name=f"bf{i}", init=OneInit
            )
            for i in range(self.arity)
        ]

    def init(self, test=False, update=True):
        # Load weights in computation graph
        self.Wi = self.Wi_p.expr(update)
        self.Wo = self.Wo_p.expr(update)
        self.Wg = self.Wg_p.expr(update)
        self.Wf = [w.expr(update) for w in self.Wf_p]
        # Load biases in computation graph
        self.bi = self.bi_p.expr(update)
        self.bo = self.bo_p.expr(update)
        self.bg = self.bg_p.expr(update)
        self.bf = [w.expr(update) for w in self.bf_p]
        # Initialize dropout mask
        self.test = test
        if not test and self.dropout > 0:
            self.dropout_mask = dy.dropout(
                dy.ones(self.incoming_dim), self.dropout)

    def _check_args_arity(self, args):
        nargs = len(args)
        # Either there is an input argument
        good_with_input = nargs == (2 * self.arity + 1)
        # Or there isn't
        good_without_input = (nargs == (2 * self.arity) and self.di == 0)
        # Otherwise raise error
        if not (good_with_input or good_without_input):
            raise ValueError(
                f"{nargs} arguments for {self.arity}-ary "
                "TreeLSTM (should be 2 * arity + 1)"
            )

    def __call__(self, *args):
        """Arguments are of the form h1, c1, h2, c2, ..., h_N, c_N[, x]"""
        self._check_args_arity(args)
        # Concatenate inputs
        h_m1 = args[::2]
        c_m1 = args[1::2]
        h_m1 = dy.concatenate(list(h_m1))
        # Dropout
        if not self.test and self.dropout > 0:
            h_m1 = dy.cmult(self.dropout_mask, h_m1)
        # Compute gates
        self.i = dy.logistic(dy.affine_transform([self.bi, self.Wi, h_m1]))
        self.o = dy.logistic(dy.affine_transform([self.bo, self.Wo, h_m1]))
        self.g = dy.tanh(dy.affine_transform([self.bg, self.Wg, h_m1]))
        # Forget gates are computed differently if the connections are diagonal
        if self.diagonal:
            self.f = [dy.logistic(dy.affine_transform(
                [self.bf[i], self.Wf[i], h_m1[i]])) for i in range(self.arity)]
        else:
            self.f = [dy.logistic(dy.affine_transform(
                [self.bf[i], self.Wf[i], h_m1])) for i in range(self.arity)]
        # Update c with gating
        self.c = dy.cmult(self.i, self.g)
        if self.arity > 0:
            self.c += dy.esum([dy.cmult(self.f[i], c_m1[i])
                               for i in range(self.arity)])
        # Output
        self.h = dy.cmult(self.o, dy.tanh(self.c))
        # Return h and c
        return self.h, self.c


def LSTM(di, dh, pc, dropout=0.0):
    """Helper function for the standard LSTM"""
    return NaryTreeLSTMCell(1, di, dh, pc, dropout=dropout, diagonal=False)


def BinaryTreeLSTM(di, dh, pc, dropout=0.0, diagonal=False):
    """Helper function for the standard Binary tree-lstm"""
    return NaryTreeLSTMCell(2, di, dh, pc, dropout=dropout, diagonal=diagonal)


class CompactLSTM(BaseLayer):
    """Standard LSTM using dynet"s memory efficient LSTM operations"""

    def __init__(self, di, dh, pc, dropout=0.0):
        super(CompactLSTM, self).__init__(pc, "compact-lstm")
        self.di = di
        self.dh = dh
        self.dropout = dropout

        # Parameters
        scale = np.sqrt(6 / (2 * self.dh + self.di))
        self.Whx_p = self.pc.add_parameters(
            (self.dh * 4, self.di),
            name="Whx",
            init=dy.UniformInitializer(scale)
        )
        self.Whh_p = self.pc.add_parameters(
            (self.dh * 4, self.dh),
            name="Whh",
            init=dy.UniformInitializer(scale)
        )
        self.bh_p = self.pc.add_parameters(
            (self.dh * 4,), name="bh", init=ZeroInit
        )

    def init(self, test=False, update=True):
        # Load weights in computation graph
        self.Whx = self.Whx_p.expr(update)
        self.Whh = self.Whh_p.expr(update)
        self.bh = self.bh_p.expr(update)
        # Initialize dropout mask
        self.test = test
        if not test and self.dropout > 0:
            self.dropout_mask_x = dy.dropout(dy.ones(self.di), self.dropout)
            self.dropout_mask_h = dy.dropout(dy.ones(self.dh), self.dropout)

    def __call__(self, h, c, x):
        if not self.test and self.dropout > 0:
            gates = dy.vanilla_lstm_gates_dropout(
                x,
                h,
                self.Whx,
                self.Whh,
                self.bh,
                self.dropout_mask_x,
                self.dropout_mask_h
            )
        else:
            gates = dy.vanilla_lstm_gates(x, h, self.Whx, self.Whh, self.bh)
        new_c = dy.vanilla_lstm_c(c, gates)
        new_h = dy.vanilla_lstm_h(new_c, gates)
        return new_h, new_c


class StackedLSTM(BaseLayer):
    """Wrapper for stacked LSTM layers

    Takes in a sequence of LSTM cells
    """

    def __init__(self, *args):
        self.layers = args

    def init(self, test=False, update=True):
        for layer in self.layers:
            layer.init(test=test, update=update)

    def __call__(self, *args):
        """Arguments are of the form h1, c1, h2, c2,...,h_N, c_N, x"""
        x = args[-1]
        hs = args[:-1:2]
        cs = args[1::2]

        outputs = []
        for i, (h, c, layer) in enumerate(zip(hs, cs, self.layers)):
            if i == 0:
                new_h, new_c = layer(h, c, x)
            else:
                new_h, new_c = layer(h, c, outputs[-2])
            outputs.append(new_h)
            outputs.append(new_c)
        return outputs


def transduce_lstm(lstm, xs, h0, c0, lengths=None, backward=False):
    """Helper function for LSTM transduction with masking

    Args:
        lstm ([type]): LSTM cell
        xs ([type]): List of input expressions
        h0 ([type]): Initial h state
        c0 ([type]): Initial c state
        lengths ([type], optional): List of lengths of all the input sequences
            in the minibatch (for masking)
        backward (bool, optional): Do backward transduction
            (with appropriate masking)

    Returns:
        list: lsit of output state (in the same order as the input)
    """

    h, c = h0, c0
    hs = []
    batch_size = xs[0].dim()[-1]
    if backward:
        xs = xs[::-1]
    if lengths is not None:
        for i, x in enumerate(xs):
            h, c = lstm(h, c, x)
            hs.append(h)
            # Compute mask
            if backward:
                m = (np.full(batch_size, i) >= (len(xs) - lengths)).astype(int)
            else:
                m = (np.full(batch_size, i) < lengths).astype(int)
            m_e = dy.inputTensor(m, batched=True)
            h, c = dy.cmult(h, m_e), dy.cmult(c, m_e)
    else:
        for x in xs:
            h, c = lstm(h, c, x)
            hs.append(h)
    if backward:
        hs = hs[::-1]
    return hs


def transduce_bilstm(
    lstm_forward,
    lstm_backward,
    xs,
    h0_fwd,
    c0_fwd,
    h0_bwd,
    c0_bwd,
    lengths=None
):
    """Helper function for biLSTM transduction with masking

    Args:
        lstm_forward ([type]): LSTM cell to be used in the forward pass
        lstm_backward ([type]): LSTM cell to be used in the backward pass
        xs (list): List of input expressions
        h0_fwd (dynet.Expression): Initial state h of the forward pass
        c0_fwd (dynet.Expression): Initial state c of the forward pass
        h0_bwd (dynet.Expression): Initial state h of the backward pass
        c0_bwd (dynet.Expression): Initial state c of the backward pass
        lengths (list, optional): List of lengths of all the input sequences
            in the minibatch (for masking)

    Returns:
        tuple: hs_fwd, hs_bwd: list of output states of the forward
            and backward pass
    """

    hs_fwd = transduce_lstm(
        lstm_forward, xs, h0_fwd, c0_fwd, lengths=lengths, backward=False
    )
    hs_bwd = transduce_lstm(
        lstm_backward, xs, h0_bwd, c0_bwd, lengths=lengths, backward=True
    )
    return hs_fwd, hs_bwd
