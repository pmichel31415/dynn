#!/usr/bin/env python3
"""
Normalization layers
====================
"""
import dynet as dy

from ..operations import unsqueeze
from ..parameter_initialization import ZeroInit, OneInit
from .base_layers import ParametrizedLayer


class LayerNorm(ParametrizedLayer):
    """Layer normalization layer:

    :math:`y=\\frac{g}{\sigma(x)}\cdot(x-\mu(x)+b)`

    Args:
        input_dim (int, tuple): Input dimension
        pc (:py:class:`dynet.ParameterCollection`): Parameter collection to
            hold the parameters
    """

    def __init__(self, pc, input_dim):
        super(LayerNorm, self).__init__(pc, "layer-norm")
        # Hyperparameters
        self.input_dim = input_dim
        # Initialize bias and gain parameters
        self.gain_p = self.pc.add_parameters(
            input_dim, name="gain", init=OneInit())
        self.bias_p = self.pc.add_parameters(
            input_dim, name="bias", init=ZeroInit())

    def init(self, test=False, update=True):
        """Initialize the layer before performing computation

        Args:
            update (bool, optional): Whether to update the parameters
                (default: ``True``)
        """
        self.gain = self.gain_p.expr(update)
        self.bias = self.bias_p.expr(update)
        self.test = test

    def __call__(self, x, d=None):
        """Layer-normalize the input.

        Args:
            x (:py:class:`dynet.Expression`): Input expression

        Returns:
            :py:class:`dynet.Expression`:
                :math:`y=\\frac{g}{\sigma(x)}\cdot(x-\mu(x)+b)`
        """
        gain = self.gain
        bias = self.bias
        if d is not None:
            # Check dimension
            if len(self.input_dim) < len(x.dim()[0]):
                gain = unsqueeze(self.gain, d=d)
                bias = unsqueeze(self.bias, d=d)
            # Reduction dims
            red_dims = [dim for dim in range(len(x.dim()[0])) if dim != d]
            # Mean
            x_mean = unsqueeze(dy.mean_dim(x, d=red_dims, b=False), d=red_dims)
            # Std
            x_std = unsqueeze(dy.std_dim(x, d=red_dims, b=False), d=red_dims)
            # Whiten
            x_ = dy.cdiv(x - x_mean, x_std + 1e-12)
            # Rescale
            output = dy.cmult(x_, gain) + bias
        else:
            # Output
            output = dy.layer_norm(x, gain, bias)

        return output
