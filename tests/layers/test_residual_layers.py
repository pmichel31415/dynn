#!/usr/bin/env python3
import unittest
from unittest import TestCase

import numpy as np
import dynet as dy

from dynn import activations

from dynn.layers import functional_layers
from dynn.layers import residual_layers


class TestResidualLayer(TestCase):

    def setUp(self):
        self.layer = functional_layers.LambdaLayer(activations.tanh)
        self.shortcut_transformation = functional_layers.LambdaLayer(
            activations.sigmoid
        )
        self.layer_weight = 0.444
        self.shortcut_weight = 0.222
        self.inputs = [np.random.rand(2, 3) for _ in range(10)]

    def test_simple(self):
        # Create residual layer
        residual = residual_layers.ResidualLayer(self.layer)
        # Iterate over different inputs
        for x_val in self.inputs:
            # Initialize computation graph
            dy.renew_cg()
            # Initialize layer
            residual.init(test=False, update=True)
            # Dummy input
            x = dy.inputTensor(x_val)
            # Check value
            self.assertTrue(np.allclose(
                residual(x).npvalue(), (activations.tanh(x)+x).npvalue()
            ))

    def test_shortcut_transformation(self):
        # Create residual layer
        residual = residual_layers.ResidualLayer(
            self.layer,
            shortcut_transform=self.shortcut_transformation,
            layer_weight=self.layer_weight,
            shortcut_weight=self.shortcut_weight,
        )
        # Iterate over different inputs
        for x_val in self.inputs:
            # Initialize computation graph
            dy.renew_cg()
            # Initialize layer
            residual.init(test=False, update=True)
            # Dummy input
            x = dy.inputTensor(x_val)
            # Output
            y = residual(x)
            # Expected output
            z = self.layer_weight * activations.tanh(x) + \
                self.shortcut_weight * activations.sigmoid(x)
            # Check value
            self.assertTrue(np.allclose(y.npvalue(), z.npvalue()))


if __name__ == '__main__':
    unittest.main()
