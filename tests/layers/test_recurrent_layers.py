#!/usr/bin/env python3
import unittest
from unittest import TestCase
import numpy as np
import dynet as dy

from dynn.layers import recurrent_layers


def _test_recurrent_layer(cell, dummy_input):
    # Initialize computation graph
    dy.renew_cg()
    # Create inputs
    x = dy.inputTensor(dummy_input)
    # Initialize cell
    cell.init(test=False, update=True)
    # Run cell
    state = cell.initial_value()
    state = cell(x, *state)
    state = cell(x + 1, *state)
    # Try forward/backward
    z = dy.sum_elems(cell.get_output(state))
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


class TestStackedRecurrentCells(TestCase):

    def setUp(self):
        self.pc = dy.ParameterCollection()
        self.dims = [20, 10, 15]
        self.dropout = 0.1

    def test_lstm_lstm(self):
        sub_cells = [
            recurrent_layers.LSTM(
                self.pc,
                self.dims[0],
                self.dims[1],
                dropout_x=self.dropout,
                dropout_h=self.dropout,
            ),
            recurrent_layers.LSTM(
                self.pc,
                self.dims[1],
                self.dims[2],
                dropout_x=self.dropout,
                dropout_h=self.dropout,
            )
        ]
        # Create the stacked layer
        cell = recurrent_layers.StackedRecurrentCells(*sub_cells)
        # Test
        _test_recurrent_layer(cell, np.random.rand(self.dims[0]))

    def test_lstm_rnn(self):
        sub_cells = [
            recurrent_layers.LSTM(
                self.pc,
                self.dims[0],
                self.dims[1],
                dropout_x=self.dropout,
                dropout_h=self.dropout,
            ),
            recurrent_layers.ElmanRNN(
                self.pc,
                self.dims[1],
                self.dims[2],
                dropout=self.dropout
            )
        ]
        # Create the stacked layer
        cell = recurrent_layers.StackedRecurrentCells(*sub_cells)
        # Test
        _test_recurrent_layer(cell, np.random.rand(self.dims[0]))


if __name__ == '__main__':
    unittest.main()
