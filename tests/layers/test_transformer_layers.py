#!/usr/bin/env python3

import unittest
from unittest import TestCase

import numpy as np
import dynet as dy

from dynn import set_random_seed
from dynn.operations import seq_mask
from dynn.layers import transformer_layers

set_random_seed(14153)


class TestTransformer(TestCase):

    def setUp(self):
        self.pc = dy.ParameterCollection()
        self.nl = 3
        self.d = 8
        self.nh = 4
        self.L = 6
        self.bsz = self.L
        self.lengths = list(range(1, self.bsz + 1))
        self.dropout = 0.01

    def _test_transformer_layer(self, transform):
        # Initialize computation graph
        dy.renew_cg()
        # Create inputs
        x = dy.random_uniform((self.d, self.L), - 1, 1, batch_size=self.bsz)
        # Mask
        mask = seq_mask(self.L, self.lengths)
        # Initialize layer
        transform.init(test=False, update=True)
        # Run transformer
        y = transform(x, lengths=self.lengths)
        # Average with masking
        print(mask.npvalue())
        z_ = y * mask
        z = dy.sum_batches(z_[0])
        # Forward backward
        z.forward()
        z.backward(full=True)
        # Check dimension
        self.assertTupleEqual(y.dim()[0], (self.d, self.L))
        self.assertEqual(y.dim()[1], self.bsz)
        # Check masking
        gradients = x.gradient()
        for b, length in enumerate(self.lengths):
            grad_elem = gradients[:, :, b].T
            for pos, g_val in enumerate(grad_elem):
                is_masked = pos >= length
                zero_grad = np.allclose(g_val, 0)
                print(b, pos, is_masked)
                self.assertEqual(is_masked, zero_grad)
                self.assertTrue(not is_masked or zero_grad)

    def test_transformer_layer(self):
        # Create layer
        transform = transformer_layers.Transformer(
            self.pc,
            self.d,
            self.nh,
            dropout=self.dropout
        )
        self._test_transformer_layer(transform)

    def test_stacked_transformer_layer(self):
        # Create layer
        transform = transformer_layers.StackedTransformers(
            self.pc,
            self.nl,
            self.d,
            self.nh,
            dropout=self.dropout
        )
        self._test_transformer_layer(transform)

    def test_triu_masking(self):
        transform = transformer_layers.Transformer(
            self.pc,
            self.d,
            self.nh,
            dropout=self.dropout
        )
        for pos_loss in range(self.L):
            dy.renew_cg()
            x = dy.random_uniform((self.d, self.L), - 1,
                                  1, batch_size=self.bsz)
            # Initialize layer
            transform.init(test=False, update=True)
            # Run transformer
            y = transform(x, triu=True)
            # Sum of values at position pos_loss
            z = dy.sum_batches(y[0][pos_loss])
            # Forward backward
            z.forward()
            z.backward(full=True)
            # Check gradients
            gradients = x.gradient()
            for pos_grad in range(self.L):
                grad_at_pos = gradients[:, pos_grad, :]
                is_masked = pos_grad > pos_loss
                grad_is_zero = np.allclose(grad_at_pos, 0.0)
                self.assertEqual(is_masked, grad_is_zero)


class TestCondTransformer(TestCase):

    def setUp(self):
        self.pc = dy.ParameterCollection()
        self.nl = 3
        self.d = 8
        self.dc = 2
        self.nh = 4
        self.l_ = 7
        self.L = 6
        self.bsz = self.l_
        self.lengths = list(range(1, self.bsz + 1))
        self.dropout = 0.01

    def _test_cond_transformer(self, transform):
        # Initialize computation graph
        dy.renew_cg()
        # Create inputs
        x = dy.random_uniform((self.d, self.L), -1, 1, batch_size=self.bsz)
        c = dy.random_uniform((self.dc, self.l_), -1, 1, batch_size=self.bsz)
        # Initialize layer
        transform.init(test=False, update=True)
        # Run transformer
        y = transform(x, c, lengths_c=self.lengths, triu=True)
        # Average with masking
        z = dy.sum_batches(dy.sum_elems(y[0]))
        # Forward backward
        z.forward()
        z.backward(full=True)
        # Check dimension
        self.assertTupleEqual(y.dim()[0], (self.d, self.L))
        self.assertEqual(y.dim()[1], self.bsz)
        # Check masking
        gradients = c.gradient()
        for b, length in enumerate(self.lengths):
            grad_elem = gradients[:, :, b].T
            for pos, g_val in enumerate(grad_elem):
                is_masked = pos >= length
                zero_grad = np.allclose(g_val, 0)
                print(b, pos, is_masked)
                self.assertEqual(is_masked, zero_grad)
                self.assertTrue(not is_masked or zero_grad)

    def _test_cond_transformer_step(self, transform):
        # Initialize computation graph
        dy.renew_cg()
        # Create inputs
        x = dy.random_uniform((self.d, self.L), -1, 1, batch_size=self.bsz)
        c = dy.random_uniform((self.dc, self.l_), -1, 1, batch_size=self.bsz)
        # Initialize layer
        transform.init(test=True, update=True)
        # Run transformer
        y = transform(x, c, lengths_c=self.lengths, triu=True)
        # Now run step by step
        y_ = []
        state = None
        for i in range(self.L):
            x_i = dy.pick(x, index=i, dim=1)
            state, y_i = transform.step(state, x_i, c, lengths_c=self.lengths)
            y_.append(y_i)
        y_ = dy.concatenate(y_, d=1)
        # Average with masking
        z = dy.sum_batches(dy.squared_distance(y, y_))
        # Forward backward
        z.forward()
        z.backward(full=True)
        # Check dimension
        self.assertTupleEqual(y_.dim()[0], (self.d, self.L))
        self.assertEqual(y_.dim()[1], self.bsz)
        # Check values
        self.assertAlmostEqual(z.value(), 0.0)

    def test_cond_transformer(self):
        # Create layer
        transform = transformer_layers.CondTransformer(
            self.pc,
            self.d,
            self.dc,
            self.nh,
            dropout=self.dropout
        )
        self._test_cond_transformer(transform)

    def test_stacked_cond_transformer(self):
        # Create layer
        transform = transformer_layers.StackedCondTransformers(
            self.pc,
            self.nl,
            self.d,
            self.dc,
            self.nh,
            dropout=self.dropout
        )
        self._test_cond_transformer(transform)

    def test_cond_transformer_step(self):
        # Create layer
        transform = transformer_layers.CondTransformer(
            self.pc,
            self.d,
            self.dc,
            self.nh,
            dropout=self.dropout
        )
        self._test_cond_transformer_step(transform)

    def test_stacked_cond_transformer_step(self):
        # Create layer
        transform = transformer_layers.StackedCondTransformers(
            self.pc,
            self.nl,
            self.d,
            self.dc,
            self.nh,
            dropout=self.dropout
        )
        self._test_cond_transformer_step(transform)


if __name__ == '__main__':
    unittest.main()
