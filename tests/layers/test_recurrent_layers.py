#
import unittest
from unittest import TestCase
import numpy as np
import dynet as dy

from dynn.layers import recurrent_layers


def _test_recurrent_layer(layer, dummy_input):
    # Initialize computation graph
    dy.renew_cg()
    # Create inputs
    x = dy.inputTensor(dummy_input)
    # Initialize layer
    layer.init(test=False, update=True)
    # Run layer
    state = layer.initial_value()
    state = layer(x, *state)
    state = layer(x + 1, *state)
    # Try forward/backward
    z = dy.sum_elems(state[0])
    z.forward()
    z.backward()


class TestElmanRNN(TestCase):

    def setUp(self):
        self.pc = dy.ParameterCollection()
        self.dh = 10
        self.di = 20
        self.dropout = 0.1

    def test_lstm_layer(self):
        # Create lstm layer
        lstm = recurrent_layers.ElmanRNN(
            self.pc, self.di, self.dh, dropout=self.dropout
        )
        _test_recurrent_layer(lstm, np.random.rand(self.di))


class TestLSTM(TestCase):

    def setUp(self):
        self.pc = dy.ParameterCollection()
        self.dh = 10
        self.di = 20
        self.dropout = 0.1

    def test_lstm_layer(self):
        # Create lstm layer
        lstm = recurrent_layers.LSTM(
            self.pc,
            self.di,
            self.dh,
            dropout_x=self.dropout,
            dropout_h=self.dropout,
        )
        _test_recurrent_layer(lstm, np.random.rand(self.di))


if __name__ == '__main__':
    unittest.main()
