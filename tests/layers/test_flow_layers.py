#!/usr/bin/env python3
import unittest
from unittest import TestCase

import numpy as np
import dynet as dy
from dynn.layers import flow_layers


class TestFlatten(TestCase):

    def setUp(self):
        self.shapes = [(2,), (2, 3), (2, 3, 4)]
        self.bsz = 5

    def test_contiguous_values(self):
        # Create layer
        flatten = flow_layers.Flatten()
        # Try with varous shapes
        for shape in self.shapes:
            # Initialize computation graph
            dy.renew_cg()
            # Dummy input
            x = dy.random_normal(shape, batch_size=self.bsz)
            # Initialize layer (shouldn't do anything)
            flatten.init(test=False, update=True)
            # Run the layer
            y = flatten(x)
            # Check dimension
            self.assertEqual(len(y.dim()[0]), 1)
            self.assertEqual(y.npvalue().size, x.npvalue().size)
            self.assertEqual(y.dim()[1], self.bsz)
            # Check values as well
            self.assertTrue(np.allclose(y.vec_value(), x.vec_value()))


if __name__ == '__main__':
    unittest.main()
