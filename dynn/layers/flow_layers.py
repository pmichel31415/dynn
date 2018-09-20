#!/usr/bin/env python3
"""
Flow related layers
===================

Those layers don't perform any computation in the forward pass.
"""
import numpy as np
import dynet as dy

from .base_layers import BaseLayer


class FlattenLayer(BaseLayer):
    """Flattens the output such that there is only one dimension left
    (batch dimension notwithstanding)

    Example:

    .. code-block:: python

        # Create the layer
        flatten = FlattenLayer()
        # Dummy batched 2d input
        x = dy.zeros((3, 4), batch_size=7)
        # x.dim() -> (3, 4), 7
        y = flatten(x)
        # y.dim() -> (12,), 7


    """

    def __init__(self):
        super(FlattenLayer, self).__init__("flatten")

    def __call__(self, x):
        """Flattens the output such that there is only one dimension left
        (batch dimension notwithstanding)

        Args:
            x ([type]): [description]

        Returns:
            [type]: [description]
        """

        x_dim, batch_size = x.dim()
        # Compute the size of the flattened dimension
        new_dim = int(np.prod(x_dim))
        # Reshape
        output = dy.reshape(x, d=(new_dim,), batch_size=batch_size)
        return output
