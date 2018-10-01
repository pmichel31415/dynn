#!/usr/bin/env python3
from itertools import product
import unittest
from unittest import TestCase

import numpy as np
import dynet as dy
from dynn import util
from dynn.layers import pooling_layers


class TestPool1DLayer(TestCase):

    def setUp(self):
        self.pc = dy.ParameterCollection()
        self.N = 20
        self.di = 10
        self.bsz = 6
        self.parameters_matrix = product(
            [1, 3],  # stride
            [None, 1, 3],  # Kernel size
        )

    def _test_forward_backward(self, pool1d, stride=1, kernel_size=None):
        # Initialize computation graph
        dy.renew_cg()
        # Create inputs
        x = dy.random_uniform((self.N, self.di), -1, 1, self.bsz)
        # Initialize layer
        pool1d.init(test=False, update=True)
        # Run lstm cell
        y = pool1d(x, stride=stride, kernel_size=kernel_size)
        # Try forward/backward
        z = dy.mean_batches(dy.sum_elems(y))
        z.forward()
        z.backward()
        # Check dimensions
        full_sequence_pooling = (kernel_size or pool1d.kernel_size) is None
        kernel_size = kernel_size or pool1d.kernel_size or self.N
        out_length = self.N - kernel_size + 1
        out_length = int(np.ceil(out_length / stride))
        if full_sequence_pooling:
            expected_shape = (self.di,)
        else:
            expected_shape = (out_length, self.di)
        self.assertTupleEqual(y.dim()[0], expected_shape)
        self.assertEqual(y.dim()[1], self.bsz)

    def test_forward_backward(self):
        for stride, kernel_size in self.parameters_matrix:
            print("Testing with arguments:")
            print(f"- stride: {stride}")
            print(f"- kernel_size: {kernel_size}")
            pool1d = pooling_layers.MaxPooling1DLayer()
            self._test_forward_backward(
                pool1d, kernel_size=kernel_size, stride=stride
            )


class TestPool2DLayer(TestCase):

    def setUp(self):
        self.pc = dy.ParameterCollection()
        self.H = 20
        self.W = 15
        self.di = 10
        self.bsz = 6
        self.parameters_matrix = product(
            [None, [1, 3], [3, 1], [3, 3]],  # default stride
            [None, [3, None], [None, 3], [3, 1]],  # default kernel sizes
            [None, [1, 3], [3, 1], [3, 3]],  # stride
            [None, [3, None], [None, 3], [3, 1]],  # Kernel sizes
        )

    def _test_forward_backward(self, pool2d, strides=1, kernel_size=None):
        # Initialize computation graph
        dy.renew_cg()
        # Create inputs
        x = dy.random_uniform((self.H, self.W, self.di), -1, 1, self.bsz)
        # Initialize layer
        pool2d.init(test=False, update=True)
        # Run lstm cell
        y = pool2d(x, strides=strides, kernel_size=kernel_size)
        # Try forward/backward
        z = dy.mean_batches(dy.sum_elems(y))
        z.forward()
        z.backward()
        # Get default kernel size/pool value
        kernel_size = util._default_value(kernel_size, [None, None])
        kernel_size = [kernel_size[0] or pool2d.kernel_size[0] or self.H,
                       kernel_size[1] or pool2d.kernel_size[1] or self.W]
        strides = util._default_value(strides, [None, None])
        strides = [strides[0] or pool2d.strides[0] or 1,
                   strides[1] or pool2d.strides[1] or 1]
        # Expected height
        out_height = self.H - kernel_size[0] + 1
        out_height = int(np.ceil(out_height / strides[0]))
        # Expected width
        out_width = self.W - kernel_size[1] + 1
        out_width = int(np.ceil(out_width / strides[1]))
        # Check dimensions
        self.assertTupleEqual(y.dim()[0], (out_height, out_width, self.di))
        self.assertEqual(y.dim()[1], self.bsz)

    def test_forward_backward(self):
        for (
            default_strides,
            default_kernel_size,
            strides,
            kernel_size
        ) in self.parameters_matrix:
            print("Testing with arguments:")
            print(f"- default_strides: {default_strides}")
            print(f"- default_kernel_size: {default_kernel_size}")
            print(f"- strides: {strides}")
            print(f"- kernel_size: {kernel_size}")
            pool2d = pooling_layers.MaxPooling2DLayer(
                default_kernel_size=default_kernel_size,
                default_strides=default_strides,
            )
            self._test_forward_backward(
                pool2d, kernel_size=kernel_size, strides=strides
            )


if __name__ == '__main__':
    unittest.main()
