#!/usr/bin/env python3
import unittest
from unittest import TestCase
import dynet as dy
from dynn.layers import dense_layers


class TestDenseLayer(TestCase):

    def setUp(self):
        self.pc = dy.ParameterCollection()
        self.do = 10
        self.di = 20
        self.dropout = 0.1

    def test_forward_backward(self):
        # Create compact lstm
        dense = dense_layers.DenseLayer(
            self.pc, self.di, self.do, dropout=self.dropout
        )
        # Initialize computation graph
        dy.renew_cg()
        # Create inputs
        x = dy.random_uniform(self.di, -1, 1)
        # Initialize layer
        dense.init(test=False, update=True)
        # Run lstm cell
        y = dense(x)
        # Try forward/backward
        z = dy.sum_elems(y)
        z.forward()
        z.backward()


class TestGatedLayer(TestCase):

    def setUp(self):
        self.pc = dy.ParameterCollection()
        self.do = 10
        self.di = 20
        self.dropout = 0.1

    def test_forward_backward(self):
        # Create layer
        gated = dense_layers.GatedLayer(
            self.pc, self.di, self.do, dropout=self.dropout
        )
        # Initialize computation graph
        dy.renew_cg()
        # Create inputs
        x = dy.random_uniform(self.di, -1, 1)
        # Initialize layer
        gated.init(test=False, update=True)
        # Run lstm cell
        y = gated(x)
        # Try forward/backward
        z = dy.sum_elems(y)
        z.forward()
        z.backward()


if __name__ == '__main__':
    unittest.main()
