#!/usr/bin/env python3
from itertools import product
import unittest
from unittest import TestCase

import numpy as np
import dynet as dy

from dynn import util
from dynn.layers import convolution_layers


class TestConv1D(TestCase):

    def setUp(self):
        self.pc = dy.ParameterCollection()
        self.N = 20
        self.di = 10
        self.nk = 5
        self.kw = 3
        self.bsz = 8
        self.dropout_rate = 0.1
        self.parameters_matrix = product(
            [1, 3],  # default stride
            [True, False],  # default zero_padded
            [None, 1, 3],  # stride
            [None, True, False],  # zero_padded
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
        if zero_padded is None:
            zero_padded = conv1d.zero_padded
        out_length = self.N if zero_padded else (
            self.N - conv1d.kernel_width + 1
        )
        out_length = int(np.ceil(out_length / (stride or conv1d.stride or 1)))
        self.assertTupleEqual(y.dim()[0], (out_length, self.nk))
        self.assertEqual(y.dim()[1], self.bsz)

    def test_forward_backward(self):
        for (
            default_stride,
            default_zero_padded,
            stride,
            zero_padded,
            nobias
        ) in self.parameters_matrix:
            print("Testing with arguments:")
            print(f"- default_stride: {default_stride}")
            print(f"- default_zero_padded: {default_zero_padded}")
            print(f"- stride: {stride}")
            print(f"- zero_padded: {zero_padded}")
            print(f"- nobias: {nobias}")
            conv1d = convolution_layers.Conv1D(
                self.pc,
                self.di,
                self.nk,
                self.kw,
                dropout_rate=self.dropout_rate,
                nobias=nobias,
                zero_padded=default_zero_padded,
                stride=default_stride,
            )
            self._test_forward_backward(
                conv1d, stride=stride, zero_padded=zero_padded
            )


class TestConv2D(TestCase):

    def setUp(self):
        self.pc = dy.ParameterCollection()
        self.H = 7
        self.W = 8
        self.di = 9
        self.nk = 5
        self.ks = [2, 3]
        self.bsz = 8
        self.dropout_rate = 0.1
        self.parameters_matrix = product(
            [[1, 1], [1, 3], [4, 1], [4, 3]],  # default strides
            [True, False],  # default zero_padded
            [None, [1, 1], [1, 3], [4, 1], [4, 3]],  # strides
            [None, True, False],  # zero_padded
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
        # Retrieve expected stride
        strides = util._default_value(strides, [None, None])
        strides = [strides[0] or conv2d.strides[0] or 1,
                   strides[1] or conv2d.strides[1] or 1]
        # And padding
        zero_padded = (
            conv2d.zero_padded if zero_padded is None else zero_padded
        )
        # Check dimensions
        out_height = self.H if zero_padded else (
            self.H - conv2d.kernel_size[0] + 1
        )
        out_height = int(np.ceil(out_height / strides[0]))
        out_width = self.W if zero_padded else (
            self.W - conv2d.kernel_size[1] + 1
        )
        out_width = int(np.ceil(out_width / strides[1]))

        self.assertTupleEqual(y.dim()[0], (out_height, out_width, self.nk))
        self.assertEqual(y.dim()[1], self.bsz)

    def test_forward_backward(self):
        for (
            default_strides,
            default_zero_padded,
            strides,
            zero_padded,
            nobias
        ) in self.parameters_matrix:
            print("Testing with arguments:")
            print(f"- default_strides: {default_strides}")
            print(f"- default_zero_padded: {default_zero_padded}")
            print(f"- strides: {strides}")
            print(f"- zero_padded: {zero_padded}")
            print(f"- nobias: {nobias}")
            conv2d = convolution_layers.Conv2D(
                self.pc,
                self.di,
                self.nk,
                self.ks,
                dropout_rate=self.dropout_rate,
                nobias=nobias,
                zero_padded=default_zero_padded,
                strides=default_strides,
            )
            self._test_forward_backward(
                conv2d, strides=strides, zero_padded=zero_padded
            )


if __name__ == '__main__':
    unittest.main()
