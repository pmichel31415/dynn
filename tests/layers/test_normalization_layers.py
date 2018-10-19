#!/usr/bin/env python3

import unittest
from unittest import TestCase

import dynet as dy

from dynn.layers import normalization_layers


class TestLayerNorm(TestCase):

    def setUp(self):
        self.pc = dy.ParameterCollection()
        self.d1 = 10
        self.d2 = 20
        self.bsz = 7

    def test_single_dim(self):
        # Create compact lstm
        norm = normalization_layers.LayerNorm(self.pc, self.d1)
        # Initialize computation graph
        dy.renew_cg()
        # Create inputs
        x = dy.random_uniform(self.d1, -1, 1, batch_size=self.bsz)
        # Initialize layer
        norm.init(test=False, update=True)
        # Run lstm cell
        y = norm(x)
        # Try forward/backward
        z = dy.sum_batches(dy.sum_elems(y))
        z.forward()
        z.backward()
        # Check dim
        self.assertTupleEqual(y.dim(), ((self.d1,), self.bsz))
        # Check values
        expected_mean = norm.bias_p.as_array().mean()
        expected_std = norm.gain_p.as_array().mean()
        y_val = y.npvalue()
        for b in range(self.bsz):
            self.assertAlmostEqual(y_val[..., b].mean(), expected_mean, 6)
            self.assertAlmostEqual(y_val[..., b].std(), expected_std, 6)

    def test_broadcast(self):
        # Create compact lstm
        norm = normalization_layers.LayerNorm(self.pc, (self.d1,))
        # Initialize computation graph
        dy.renew_cg()
        # Create inputs
        x = dy.random_uniform((self.d1, self.d2), -1, 1, batch_size=self.bsz)
        # Initialize layer
        norm.init(test=False, update=True)
        # Run lstm cell
        y = norm(x, d=1)
        # Try forward/backward
        z = dy.sum_batches(dy.sum_elems(y))
        z.forward()
        z.backward()
        # Check dim
        self.assertTupleEqual(y.dim(), ((self.d1, self.d2), self.bsz))
        # Check values
        expected_mean = norm.bias_p.as_array().mean()
        expected_std = norm.gain_p.as_array().mean()
        y_val = y.npvalue()
        for b in range(self.bsz):
            for d in range(self.d2):
                self.assertAlmostEqual(y_val[:, d, b].mean(), expected_mean, 6)
                self.assertAlmostEqual(y_val[:, d, b].std(), expected_std, 6)


if __name__ == '__main__':
    unittest.main()
