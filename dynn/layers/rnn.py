#!/usr/bin/env python3
"""FIXME"""

import numpy as np
import dynet as dy

from ..parameter_initialization import ZeroInit, NormalInit
from .. import activations
from .base_layer import BaseLayer


class ElmanRNN(BaseLayer):
    """Standard Elman RNN"""

    def __init__(self, di, dh, pc, activation=activations.tanh, dropout=0.0):
        super(ElmanRNN, self).__init__(pc, 'elman-rnn')
        # Hyper parameters
        self.di = di
        self.dh = dh
        self.dropout = dropout
        self.activation = activation

        # Parameters
        scale_whx = np.sqrt(2 / (self.dh + self.di))
        scale_whh = 1.0 / np.sqrt(self.dh)
        self.Whx_p = self.pc.add_parameters(
            (self.dh, self.di), name='Whx', init=NormalInit(scale_whx))
        scale_whh = np.sqrt(1 / self.dh)
        self.Whh_p = self.pc.add_parameters(
            (self.dh, self.dh), name='Whh', init=NormalInit(scale_whh))
        self.bh_p = self.pc.add_parameters(
            (self.dh,), name='bh', init=ZeroInit)

    def init(self, test=False, update=True):
        # Load weights in computation graph
        self.Whx = self.Whx_p if update else dy.const_parameter(self.Whx_p)
        self.Whh = self.Whh_p if update else dy.const_parameter(self.Whh_p)
        self.bh = self.bh_p if update else dy.const_parameter(self.bh_p)
        # Initialize dropout mask
        self.test = test
        if not test and self.dropout > 0:
            self.dropout_mask_x = dy.dropout(dy.ones(self.di), self.dropout)
            self.dropout_mask_h = dy.dropout(dy.ones(self.dh), self.dropout)

    def __call__(self, h, c, x):
        # Dropout
        if not self.test and self.dropout > 0:
            x = dy.cmult(x, self.dropout_mask_x)
            h = dy.cmult(h, self.dropout_mask_h)
        # Compute the new hidden state
        new_h = dy.affine_transform([self.bh, self.Whh, h, self.Whx, x])
        new_h = dy.activation()
        return new_h


def transduce_rnn(lstm, xs, h0, lengths=None, backward=False):
    """Helper function for LSTM transduction with masking"""
    h = h0
    hs = []
    batch_size = xs[0].dim()[-1]
    if backward:
        xs = xs[::-1]
    if lengths is not None:
        for i, x in enumerate(xs):
            h = lstm(h, x)
            hs.append(h)
            # Compute mask
            if backward:
                m = (np.full(batch_size, i) >= (len(xs) - lengths)).astype(int)
            else:
                m = (np.full(batch_size, i) < lengths).astype(int)
            m_e = dy.inputTensor(m, batched=True)
            h = dy.cmult(h, m_e)
    else:
        for x in xs:
            h = lstm(h, x)
            hs.append(h)
    if backward:
        hs = hs[::-1]
    return hs
