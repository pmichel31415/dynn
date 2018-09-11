#!/usr/bin/env python3
"""Normalization layers"""
import dynet as dy

from dynn.layers import BaseLayer
from dynn.parameter_initialization import ZeroInit, OneInit


class LayerNormalization(BaseLayer):
    """Layer normalization layer:

    :math:`y=\\frac{g}{\sigma(x)}\cdot(x-\mu(x)+b)`

    Args:
        input_dim (int): Input dimension
        pc (dynet.ParameterCollection): Parameter collection to
            hold the parameters
    """

    def __init__(self, input_dim, pc):
        super(LayerNormalization, self).__init__(pc, "layer-norm")
        # Hyperparameters
        self.input_dim = input_dim
        # Initialize bias and gain parameters
        self.gain_p = self.pc.add_parameters(
            input_dim, name="gain", init=OneInit)
        self.bias_p = self.pc.add_parameters(
            input_dim, name="bias", init=ZeroInit)

    def init(self, update=True):
        """Initialize the layer before performing computation

        Args:
            update (bool, optional): Whether to update the parameters
                (default: ``True``)
        """
        self.gain = self.gain_p.expr(update)
        self.bias = self.bias_p.expr(update)

    def __call__(self, x):
        """Layer-normalize the input

        Args:
            x (dynet.Expression): Input expression

        Returns:
            dynet.Expression: :math:`y=\\frac{g}{\sigma(x)}\cdot(x-\mu(x)+b)`
        """

        # Output
        self.output = dy.layer_norm(x, self.gain, self.bias)
        # final output
        return self.output
