#!/usr/bin/env python3
from itertools import product
import unittest
from unittest import TestCase

import dynet as dy

from dynn.layers import dense_layers, convolution_layers, combination_layers

# For determinism
dy.reset_random_seed(31415)


class TestSequential(TestCase):

    def setUp(self):
        self.pc = dy.ParameterCollection()
        self.bsz = 6
        self.dims = [3, 4, 5, 3, 6, 5, 2, 3, 5, 4, 2]
        self.n_layers_range = range(1, len(self.dims)-1)

    def test_dense_stacking(self):
        # Create the list of dense layers
        for n_layers in self.n_layers_range:
            layers = [
                dense_layers.Affine(self.pc, self.dims[i], self.dims[i+1])
                for i in range(n_layers)
            ]
            # Stack the layers
            network = combination_layers.Sequential(*layers)
            # Run once for sanity check
            dy.renew_cg()
            # Dummy input
            x = dy.random_normal(self.dims[0], batch_size=self.bsz)
            # Init layer
            network.init(test=False, update=True)
            # Result
            y = network(x)
            # "Loss function"
            loss = dy.mean_batches(dy.sum_elems(y))
            # Forward/backward
            loss.forward()
            loss.backward()
            # Check dimensions
            self.assertTupleEqual(y.dim()[0], (self.dims[n_layers],))
            self.assertEqual(y.dim()[1], self.bsz)

    def test_empty_list(self):
        self.assertRaises(ValueError, combination_layers.Sequential)

    def test_nonlayer(self):
        self.assertRaises(
            ValueError,
            combination_layers.Sequential,
            dense_layers.Affine(self.pc, 1, 1), "Oops",
            dense_layers.Affine(self.pc, 1, 1)
        )


class TestParallel(TestCase):

    def setUp(self):
        self.pc = dy.ParameterCollection()
        self.bsz = 6
        self.di = 2
        self.H = 10
        self.W = 20
        self.ksizes = list(product([1, 2, 3], [1, 2, 3]))
        self.nk = 2
        self.n_layers_range = range(1, len(self.ksizes) - 1)

    def test_conv2d_concat(self):
        # Create the list of dense layers
        for n_layers in self.n_layers_range:
            layers = [
                convolution_layers.Conv2D(self.pc, self.di, self.nk, ksz)
                for ksz in self.ksizes[:n_layers]
            ]
            # Concatenate the layers
            network = combination_layers.Parallel(*layers, dim=-1)
            # Try both with and without insert dim
            for insert_dim in [False, True]:
                # Run once for sanity check
                dy.renew_cg()
                # Dummy input
                img = dy.random_normal(
                    (self.H, self.W, self.di), batch_size=self.bsz
                )
                # Init layer
                network.init(test=False, update=True)
                # Result
                y = network(img, insert_dim=insert_dim)
                # "Loss function"
                loss = dy.mean_batches(dy.sum_elems(y))
                # Forward/backward
                loss.forward()
                loss.backward()
                # Check dimensions
                if insert_dim:
                    self.assertTupleEqual(
                        y.dim()[0], (self.H, self.W, self.nk, n_layers)
                    )
                else:
                    self.assertTupleEqual(
                        y.dim()[0], (self.H, self.W, self.nk * n_layers)
                    )
                self.assertEqual(y.dim()[1], self.bsz)

    def test_empty_list(self):
        self.assertRaises(ValueError, combination_layers.Parallel)

    def test_nonlayer(self):
        self.assertRaises(
            ValueError,
            combination_layers.Parallel,
            dense_layers.Affine(self.pc, 1, 1), "Oops",
            dense_layers.Affine(self.pc, 1, 1)
        )


if __name__ == '__main__':
    unittest.main()
