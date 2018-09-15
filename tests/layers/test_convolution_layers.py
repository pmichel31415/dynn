#!/usr/bin/env python3
from itertools import product
import unittest
from unittest import TestCase

import numpy as np
import dynet as dy
from dynn.layers import convolution_layers


class TestConv1DLayer(TestCase):

    def setUp(self):
        self.pc = dy.ParameterCollection()
        self.N = 20
        self.di = 10
        self.nk = 5
        self.kw = 3
        self.bsz = 8
        self.dropout_rate = 0.1
        self.parameters_matrix = product(
            [1, 3],  # stride
            [True, False],  # zero_padded
            [False, True],  # nobias
        )

    def _test_forward_backward(self, conv1d, stride=1, zero_padded=True):
        # Initialize computation graph
        dy.renew_cg()
        # Create inputs
        x = dy.random_uniform((self.N, self.di), -1, 1, self.bsz)
        # Initialize layer
        conv1d.init(test=False, update=True)
        # Run lstm cell
        y = conv1d(x, stride=stride, zero_padded=zero_padded)
        # Try forward/backward
        z = dy.mean_batches(dy.sum_elems(y))
        z.forward()
        z.backward()
        # Check dimensions
        out_length = self.N if zero_padded else (
            self.N - conv1d.kernel_width + 1
        )
        out_length = int(np.ceil(out_length / stride))
        self.assertTupleEqual(y.dim()[0], (out_length, self.nk))
        self.assertEqual(y.dim()[1], self.bsz)

    def test_forward_backward(self):
        for stride, zero_padded, nobias in self.parameters_matrix:
            print("Testing with arguments:")
            print(f"- stride: {stride}")
            print(f"- zero_padded: {zero_padded}")
            print(f"- nobias: {nobias}")
            conv1d = convolution_layers.Conv1DLayer(
                self.pc,
                self.di,
                self.nk,
                self.kw,
                dropout_rate=self.dropout_rate,
                nobias=nobias,
            )
            self._test_forward_backward(
                conv1d, stride=stride, zero_padded=zero_padded
            )


class TestConv2DLayer(TestCase):

    def setUp(self):
        self.pc = dy.ParameterCollection()
        self.H = 20
        self.W = 15
        self.di = 10
        self.nk = 5
        self.kh = 4
        self.kw = 3
        self.bsz = 8
        self.dropout_rate = 0.1
        self.parameters_matrix = product(
            [[1, 1], [1, 3], [4, 1], [4, 3]],  # strides
            [True, False],  # zero_padded
            [False, True],  # nobias
        )

    def _test_forward_backward(self, conv2d, strides=1, zero_padded=True):
        # Initialize computation graph
        dy.renew_cg()
        # Create inputs
        x = dy.random_uniform((self.H, self.W, self.di), -1, 1, self.bsz)
        # Initialize layer
        conv2d.init(test=False, update=True)
        # Run lstm cell
        y = conv2d(x, strides=strides, zero_padded=zero_padded)
        # Try forward/backward
        z = dy.mean_batches(dy.sum_elems(y))
        z.forward()
        z.backward()
        # Check dimensions
        out_height = self.H if zero_padded else (
            self.H - conv2d.kernel_height + 1
        )
        out_height = int(np.ceil(out_height / strides[0]))

        out_width = self.W if zero_padded else (
            self.W - conv2d.kernel_width + 1
        )
        out_width = int(np.ceil(out_width / strides[1]))

        self.assertTupleEqual(y.dim()[0], (out_height, out_width, self.nk))
        self.assertEqual(y.dim()[1], self.bsz)

    def test_forward_backward(self):
        for strides, zero_padded, nobias in self.parameters_matrix:
            print("Testing with arguments:")
            print(f"- strides: {strides}")
            print(f"- zero_padded: {zero_padded}")
            print(f"- nobias: {nobias}")
            conv2d = convolution_layers.Conv2DLayer(
                self.pc,
                self.di,
                self.nk,
                self.kh,
                self.kw,
                dropout_rate=self.dropout_rate,
                nobias=nobias,
            )
            self._test_forward_backward(
                conv2d, strides=strides, zero_padded=zero_padded
            )


if __name__ == '__main__':
    unittest.main()
