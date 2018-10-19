#!/usr/bin/env python3

import unittest
from unittest import TestCase

import numpy as np
import dynet as dy

from dynn import set_random_seed
from dynn.operations import seq_mask
from dynn.layers import transformer_layers

set_random_seed(14153)


class TestTransformerLayer(TestCase):

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
        transform = transformer_layers.TransformerLayer(
            self.pc,
            self.d,
            self.nh,
            dropout=self.dropout
        )
        self._test_transformer_layer(transform)

    def test_stacked_transformer_layer(self):
        # Create layer
        transform = transformer_layers.StackedTransformerLayers(
            self.pc,
            self.nl,
            self.d,
            self.nh,
            dropout=self.dropout
        )
        self._test_transformer_layer(transform)


if __name__ == '__main__':
    unittest.main()
