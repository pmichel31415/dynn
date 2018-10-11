#!/usr/bin/env python3
from itertools import product
import unittest
from unittest import TestCase

import numpy as np
import dynet as dy

from dynn.layers import dense_layers, recurrent_layers, transduction_layers


class TestTransduction(TestCase):

    def setUp(self):
        self.pc = dy.ParameterCollection()
        self.do = 10
        self.di = 20
        self.bz = 6

    def test_feedforward_layer_transduction(self):
        # Simple dense layer
        dense = dense_layers.Affine(self.pc, self.di, self.do)
        # Create transduction layer
        tranductor = transduction_layers.Transduction(dense)
        # Initialize computation graph
        dy.renew_cg()
        # Create inputs
        seq = [dy.random_uniform(self.di, 0, i, self.bz) for i in range(10)]
        # Initialize tranductor
        tranductor.init(test=False, update=True)
        # Run tranductor
        outputs = tranductor(seq)
        # Try forward/backward
        z = dy.mean_batches(dy.sum_elems(dy.esum(outputs)))
        z.forward()
        z.backward()


class TestSequenceMaskingLayer(TestCase):

    def setUp(self):
        self.pc = dy.ParameterCollection()
        self.do = 10
        self.di = 20
        self.bz = 6
        self.lengths = [1, 2, 3, 4, 5, 6]
        self.mask_value = 42.0

    def test_left_padded(self):
        # Create transduction layer
        tranductor = transduction_layers.SequenceMaskingLayer(
            mask_value=self.mask_value)
        # Initialize computation graph
        dy.renew_cg()
        # Create inputs
        seq = [dy.random_uniform(self.di, 0, i, self.bz) for i in range(6)]
        # Initialize tranductor
        tranductor.init(test=False, update=True)
        # Run tranductor
        outputs = tranductor(seq, self.lengths)
        # Try forward/backward
        z = dy.mean_batches(dy.sum_elems(dy.esum(outputs)))
        z.forward()
        z.backward(full=True)
        # Check value
        for idx, length in enumerate(self.lengths):
            for step in range(length, len(seq)):
                values = outputs[step].npvalue()[:, idx]
                for value in values:
                    self.assertAlmostEquals(value, self.mask_value, 10)
        # Check gradients
        for idx, length in enumerate(self.lengths):
            for step in range(length, len(seq)):
                grad = seq[step].gradient()[:, idx]
                self.assertAlmostEquals(np.abs(grad).sum(), 0, 10)

    def test_right_padded(self):
        # Create transduction layer
        tranductor = transduction_layers.SequenceMaskingLayer(
            mask_value=self.mask_value, left_padded=False)
        # Initialize computation graph
        dy.renew_cg()
        # Create inputs
        seq = [dy.random_uniform(self.di, 0, i, self.bz) for i in range(6)]
        # Initialize tranductor
        tranductor.init(test=False, update=True)
        # Run tranductor
        outputs = tranductor(seq, self.lengths)
        # Try forward/backward
        z = dy.mean_batches(dy.sum_elems(dy.esum(outputs)))
        z.forward()
        z.backward(full=True)
        # Check dimensions
        for x, state in zip(seq, outputs):
            for s in state:
                self.assertEqual(x.dim()[1], s.dim()[1])
        # Check value
        for idx, length in enumerate(self.lengths):
            for step in range(len(seq)-length):
                values = outputs[step].npvalue()[:, idx]
                for value in values:
                    self.assertAlmostEquals(value, self.mask_value, 10)
        value = outputs[0].npvalue()[0, 0]
        self.assertAlmostEquals(value, self.mask_value, 10)
        # Check gradients
        for idx, length in enumerate(self.lengths):
            for step in range(len(seq)-length):
                grad = seq[step].gradient()[:, idx]
                self.assertAlmostEquals(np.abs(grad).sum(), 0, 10)


class TestUnidirectional(TestCase):

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

    def _test_recurrent_layer_unidirectional_transduction(
        self,
        layer,
        dummy_input,
        lengths,
        backward,
        left_padded,
    ):
        # Create transduction layer
        tranductor = transduction_layers.Unidirectional(layer)
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
        z = dy.mean_batches(
            dy.esum([dy.sum_elems(state[0]) for state in states]))
        z.forward()
        z.backward(full=True)
        # check masking
        if lengths is not None:
            for idx, length in enumerate(lengths):
                if left_padded:
                    masked_steps = range(length, len(seq))
                else:
                    masked_steps = range(len(seq)-length)
                # Values
                for step in masked_steps:
                    for state in states[step]:
                        values = state.npvalue()[:, idx]
                        for value in values:
                            self.assertAlmostEquals(value, 0, 10)
                    # Check gradients
                    grad = seq[step].gradient()[:, idx]
                    self.assertAlmostEquals(np.abs(grad).sum(), 0, 10)

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
            self._test_recurrent_layer_unidirectional_transduction(
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
            self._test_recurrent_layer_unidirectional_transduction(
                lstm,
                np.random.rand(self.di, self.bz),
                lengths,
                backward,
                left_padded
            )

    def test_stacked_lstm_rnn(self):
        # Create lstm layer
        cells = [
            recurrent_layers.LSTM(
                self.pc,
                self.di,
                self.dh-1,
                dropout_x=self.dropout,
                dropout_h=self.dropout,
            ),
            recurrent_layers.ElmanRNN(
                self.pc,
                self.dh-1,
                self.dh,
                dropout=self.dropout,
            ),
        ]
        stacked_cell = recurrent_layers.StackedRecurrentCells(*cells)
        for lengths, backward, left_padded in self.parameters_matrix:
            print(f"Testing with:")
            print(f"- lengths=: {lengths}")
            print(f"- backward=: {backward}")
            print(f"- left_padded=: {left_padded}")
            self._test_recurrent_layer_unidirectional_transduction(
                stacked_cell,
                np.random.rand(self.di, self.bz),
                lengths,
                backward,
                left_padded
            )


class TestBidirectional(TestCase):

    def setUp(self):
        self.pc = dy.ParameterCollection()
        self.dh = 10
        self.di = 20
        self.bz = 6
        self.dropout = 0.1
        self.parameters_matrix = product(
            [None, [1, 2, 3, 4, 5, 6], [4, 5, 6, 6, 1, 2]],  # lengths
            [True, False],  # left_padded
        )

    def _test_recurrent_layer_bidirectional_transduction(
        self,
        fwd_layer,
        bwd_layer,
        dummy_input,
        lengths,
        left_padded,
    ):
        # Create transduction layer
        tranductor = transduction_layers.Bidirectional(
            fwd_layer, bwd_layer)
        # Initialize computation graph
        dy.renew_cg()
        # Create inputs
        seq = [
            dy.inputTensor(dummy_input, batched=True) + i for i in range(10)
        ]
        # Initialize tranductor
        tranductor.init(test=False, update=True)
        # Run tranductor
        fwd_states, bwd_states = tranductor(
            seq, lengths=lengths, left_padded=left_padded
        )
        # Try forward/backward
        fwd_z = dy.mean_batches(
            dy.esum([dy.sum_elems(state[0]) for state in fwd_states])
        )
        bwd_z = dy.mean_batches(
            dy.esum([dy.sum_elems(state[0]) for state in bwd_states])
        )
        z = fwd_z + bwd_z
        z.forward()
        z.backward()
        # Check dimensions
        for x, state in zip(seq, fwd_states):
            for s in state:
                self.assertEqual(x.dim()[1], s.dim()[1])
        for x, state in zip(seq, bwd_states):
            for s in state:
                self.assertEqual(x.dim()[1], s.dim()[1])
        # check masking
        if lengths is not None:
            for idx, length in enumerate(lengths):
                if left_padded:
                    masked_steps = range(length, len(seq))
                else:
                    masked_steps = range(len(seq)-length)
                # Values
                for step in masked_steps:
                    for state in fwd_states[step] + bwd_states[step]:
                        values = state.npvalue()[:, idx]
                        for value in values:
                            self.assertAlmostEquals(value, 0, 10)
                    # Check gradients
                    grad = seq[step].gradient()[:, idx]
                    self.assertAlmostEquals(np.abs(grad).sum(), 0, 10)

    def test_bi_elman_rnn(self):
        # Create rnn layers
        fwd_rnn = recurrent_layers.ElmanRNN(
            self.pc, self.di, self.dh, dropout=self.dropout
        )
        bwd_rnn = recurrent_layers.ElmanRNN(
            self.pc, self.di, self.dh, dropout=self.dropout
        )
        for lengths, left_padded in self.parameters_matrix:
            print(f"Testing with:")
            print(f"- lengths=: {lengths}")
            print(f"- left_padded=: {left_padded}")
            self._test_recurrent_layer_bidirectional_transduction(
                fwd_rnn,
                bwd_rnn,
                np.random.rand(self.di, self.bz),
                lengths,
                left_padded
            )

    def test_bi_lstm(self):
        # Create lstm layers
        fwd_lstm = recurrent_layers.LSTM(
            self.pc,
            self.di,
            self.dh,
            dropout_x=self.dropout,
            dropout_h=self.dropout,
        )
        bwd_lstm = recurrent_layers.LSTM(
            self.pc,
            self.di,
            self.dh,
            dropout_x=self.dropout,
            dropout_h=self.dropout,
        )

        for lengths, left_padded in self.parameters_matrix:
            print(f"Testing with:")
            print(f"- lengths=: {lengths}")
            print(f"- left_padded=: {left_padded}")
            self._test_recurrent_layer_bidirectional_transduction(
                fwd_lstm,
                bwd_lstm,
                np.random.rand(self.di, self.bz),
                lengths,
                left_padded
            )

    def test_rnn_lstm(self):
        # Create rnn/lstm layers
        fwd_lstm = recurrent_layers.LSTM(
            self.pc,
            self.di,
            self.dh,
            dropout_x=self.dropout,
            dropout_h=self.dropout,
        )
        bwd_rnn = recurrent_layers.ElmanRNN(
            self.pc, self.di, self.dh, dropout=self.dropout
        )
        for lengths, left_padded in self.parameters_matrix:
            print(f"Testing with:")
            print(f"- lengths=: {lengths}")
            print(f"- left_padded=: {left_padded}")
            self._test_recurrent_layer_bidirectional_transduction(
                fwd_lstm,
                bwd_rnn,
                np.random.rand(self.di, self.bz),
                lengths,
                left_padded
            )


if __name__ == '__main__':
    unittest.main()
