#!/usr/bin/env python3
import unittest
from unittest import TestCase

import numpy as np
import dynet as dy
from dynn.layers import functional_layers


class TestConstantLayer(TestCase):

    def setUp(self):
        self.scalar = 5
        self.list = [5, 6]
        self.nparray = np.random.rand(10, 12)

    def test_scalar(self):
        # Create constant layer
        constant = functional_layers.ConstantLayer(self.scalar)
        # Initialize computation graph
        dy.renew_cg()
        # Initialize layer
        constant.init(test=False, update=True)
        # Run layer
        y = constant()
        # Check value
        self.assertEqual(y.scalar_value(), self.scalar)

    def test_list(self):
        # Create constant layer
        constant = functional_layers.ConstantLayer(self.list)
        # Initialize computation graph
        dy.renew_cg()
        # Initialize layer
        constant.init(test=False, update=True)
        # Run layer
        y = constant()
        z = dy.sum_elems(y)
        # Try forward/backward
        z.forward()
        z.backward()
        # Check value
        self.assertListEqual(y.vec_value(), self.list)

    def test_nparray(self):
        # Create constant layer
        constant = functional_layers.ConstantLayer(self.nparray)
        # Initialize computation graph
        dy.renew_cg()
        # Initialize layer
        constant.init(test=False, update=True)
        # Run layer
        y = constant()
        z = dy.sum_elems(y)
        # Try forward/backward
        z.forward()
        z.backward()
        # Check value
        self.assertTrue(np.allclose(y.npvalue(), self.nparray))


class TestUnaryOpLayer(TestCase):

    def setUp(self):
        self.scalar = 42
        self.layer = functional_layers.ConstantLayer(self.scalar)

    def test_neg(self):
        # Create negation layers
        neg_layer_1 = functional_layers.NegationLayer(self.layer)
        neg_layer_2 = - self.layer
        # Initialize computation graph
        dy.renew_cg()
        # Initialize layer
        neg_layer_1.init(test=False, update=True)
        neg_layer_2.init(test=False, update=True)
        # Run layer
        y_1 = neg_layer_1()
        y_2 = neg_layer_2()
        # Try forward/backward
        y_1.forward()
        y_2.forward()
        y_1.backward()
        y_2.backward()
        # Check value
        self.assertEqual(y_1.scalar_value(), -self.scalar)
        self.assertEqual(y_2.scalar_value(), -self.scalar)

    def test_identity(self):
        # Create identity layer
        identity = functional_layers.IdentityLayer(self.layer)
        # Initialize computation graph
        dy.renew_cg()
        # Initialize layer
        identity.init(test=False, update=True)
        # Run layer
        y = identity()
        # Try forward/backward
        y.forward()
        y.backward()
        # Check value
        self.assertEqual(y.scalar_value(), self.scalar)


class TestBinaryOpLayer(TestCase):

    def setUp(self):
        self.scalar1 = 42
        self.layer1 = functional_layers.ConstantLayer(self.scalar1)

        self.scalar2 = 25
        self.layer2 = functional_layers.ConstantLayer(self.scalar2)

    def test_add(self):
        # Create negation layers
        add_layer_1 = functional_layers.AdditionLayer(self.layer1, self.layer2)
        add_layer_2 = self.layer1 + self.layer2
        add_layer_3 = self.layer2 + self.layer1
        # Initialize computation graph
        dy.renew_cg()
        # Initialize layer
        add_layer_1.init(test=False, update=True)
        add_layer_2.init(test=False, update=True)
        add_layer_3.init(test=False, update=True)
        # Run layer
        y_1 = add_layer_1()
        y_2 = add_layer_2()
        y_3 = add_layer_3()
        # Try forward/backward
        y_1.forward()
        y_2.forward()
        y_3.forward()
        y_1.backward()
        y_2.backward()
        y_3.backward()
        # Check value
        self.assertEqual(y_1.scalar_value(), self.scalar1+self.scalar2)
        self.assertEqual(y_2.scalar_value(), self.scalar1+self.scalar2)
        self.assertEqual(y_3.scalar_value(), self.scalar1+self.scalar2)

    def test_cmult(self):
        # Create cmult layer
        cmult = functional_layers.CmultLayer(self.layer1, self.layer2)
        # Initialize computation graph
        dy.renew_cg()
        # Initialize layer
        cmult.init(test=False, update=True)
        # Run layer
        y = cmult()
        # Try forward/backward
        y.forward()
        y.backward()
        # Check value
        self.assertEqual(y.scalar_value(), self.scalar1*self.scalar2)


if __name__ == '__main__':
    unittest.main()
