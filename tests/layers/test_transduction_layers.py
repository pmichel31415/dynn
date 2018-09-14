#!/usr/bin/env python3
from itertools import product
import unittest
from unittest import TestCase

import numpy as np
import dynet as dy

from dynn.layers import recurrent_layers, transduction_layers


def _test_recurrent_layer_transduction(
    layer,
    dummy_input,
    lengths,
    backward,
    left_padded,
):
    # Create transduction layer
    tranductor = transduction_layers.UnidirectionalLayer(layer)
    # Initialize computation graph
    dy.renew_cg()
    # Create inputs
    seq = [
        dy.inputTensor(dummy_input, batched=True) + i for i in range(10)
    ]
    # Initialize tranductor
    tranductor.init(test=False, update=True)
    # Run tranductor
    states = tranductor(
        seq, lengths=lengths, backward=backward, left_padded=left_padded
    )
    # Try forward/backward
    z = dy.mean_batches(dy.esum([dy.sum_elems(state[0]) for state in states]))
    z.forward()
    z.backward()


class TestUnidirectionalLayer(TestCase):

    def setUp(self):
        self.pc = dy.ParameterCollection()
        self.dh = 10
        self.di = 20
        self.bz = 6
        self.dropout = 0.1
        self.parameters_matrix = product(
            [None, [1, 2, 3, 4, 5, 6], [4, 5, 6, 6, 1, 2]],  # lengths
            [False, True],  # backward
            [True, False],  # left_padded
        )

    def test_elman_rnn(self):
        # Create lstm layer
        lstm = recurrent_layers.ElmanRNN(
            self.pc, self.di, self.dh, dropout=self.dropout
        )
        for lengths, backward, left_padded in self.parameters_matrix:
            print(f"Testing with:")
            print(f"- lengths=: {lengths}")
            print(f"- backward=: {backward}")
            print(f"- left_padded=: {left_padded}")
            _test_recurrent_layer_transduction(
                lstm,
                np.random.rand(self.di, self.bz),
                lengths,
                backward,
                left_padded
            )

    def test_lstm(self):
        # Create lstm layer
        lstm = recurrent_layers.LSTM(
            self.pc,
            self.di,
            self.dh,
            dropout_x=self.dropout,
            dropout_h=self.dropout,
        )
        for lengths, backward, left_padded in self.parameters_matrix:
            print(f"Testing with:")
            print(f"- lengths=: {lengths}")
            print(f"- backward=: {backward}")
            print(f"- left_padded=: {left_padded}")
            _test_recurrent_layer_transduction(
                lstm,
                np.random.rand(self.di, self.bz),
                lengths,
                backward,
                left_padded
            )


if __name__ == '__main__':
    unittest.main()
