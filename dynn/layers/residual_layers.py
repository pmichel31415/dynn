#!/usr/bin/env python3
"""
Residual layers
===============
"""

from ..util import _default_value

from .base_layers import BaseLayer
from .functional_layers import IdentityLayer


class ResidualLayer(BaseLayer):
    """Adds residual connections to a layer"""

    def __init__(
        self,
        layer,
        residual_transform=None,
        layer_weight=1.0,
        residual_weight=1.0,
    ):
        super(ResidualLayer, self).__init__("residual")
        self.layer = layer
        self.residual_transform = _default_value(
            residual_transform, IdentityLayer()
        )
        self.layer_weight = layer_weight
        self.residual_weight = residual_weight

    def init(self, *args, **kwargs):
        """Initialize the layer before performing computation

        For example setup dropout, freeze some parameters, etc...
        """
        self.layer.init(*args, **kwargs)
        self.residual_transform.init(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        """Execute forward pass"""
        output = self.layer_weight * self.layer(*args, **kwargs)
        residual_connections = self.residual_weight * \
            self.residual_transform(*args, **kwargs)
        return output + residual_connections
