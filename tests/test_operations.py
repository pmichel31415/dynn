#!/usr/bin/env python3

import unittest
from unittest import TestCase

import numpy as np
import dynet as dy

from dynn import operations


class TestOperations(TestCase):

    def setUp(self):
        self.to_squeeze = [(1, 2, 3), (2, 1, 3), (2, 3, 1)]
        self.squeezed = (2, 3)
        self.n_stack = 5
        self.to_stack = [(self.n_stack, 2, 3),
                         (2, self.n_stack, 3),
                         (2, 3, self.n_stack)]

    def test_squeeze(self):
        # Test on various shapes
        for dim, shape in enumerate(self.to_squeeze):
            dy.renew_cg()
            # Dummy input
            x = dy.random_normal(shape)
            # Squeeze the dim
            y = operations.squeeze(x, d=dim)
            # Check shape
            self.assertTupleEqual(y.dim()[0], self.squeezed)
            # Check values
            self.assertTrue(np.allclose(x.vec_value(), y.vec_value()))
        # Try with negative dim values
        for dim, shape in enumerate(self.to_squeeze):
            dy.renew_cg()
            # Dummy input
            x = dy.random_normal(shape)
            # Squeeze the dim
            y = operations.squeeze(x, d=dim-3)
            # Check shape
            self.assertTupleEqual(y.dim()[0], self.squeezed)
            # Check values
            self.assertTrue(np.allclose(x.vec_value(), y.vec_value()))

    def test_unsqueeze(self):
        # Test on various shapes
        for dim, shape in enumerate(self.to_squeeze):
            dy.renew_cg()
            # Dummy input
            x = dy.random_normal(self.squeezed)
            # Squeeze the dim
            y = operations.unsqueeze(x, d=dim)
            # Check shape
            self.assertTupleEqual(y.dim()[0], shape)
            # Check values
            self.assertTrue(np.allclose(x.vec_value(), y.vec_value()))
        # Try with negative dim values
        for dim, shape in enumerate(self.to_squeeze):
            dy.renew_cg()
            # Dummy input
            x = dy.random_normal(self.squeezed)
            # Squeeze the dim
            y = operations.unsqueeze(x, d=dim-3)
            # Check shape
            self.assertTupleEqual(y.dim()[0], shape)
            # Check values
            self.assertTrue(np.allclose(x.vec_value(), y.vec_value()))

    def test_stack(self):
        # Test on various shapes
        for dim, shape in enumerate(self.to_stack):
            dy.renew_cg()
            # Dummy input
            xs = [dy.random_normal(self.squeezed) for _ in range(self.n_stack)]
            # Stack
            y = operations.stack(xs, d=dim)
            # Check shape
            self.assertTupleEqual(y.dim()[0], shape)
            # Check values
            xs_val = [x.npvalue() for x in xs]
            y_val = y.npvalue()
            slices = [slice(None), slice(None), slice(None)]
            for i, x_val in enumerate(xs_val):
                slices[dim] = i
                self.assertTrue(np.allclose(x_val, y_val[tuple(slices)]))


if __name__ == '__main__':
    unittest.main()
