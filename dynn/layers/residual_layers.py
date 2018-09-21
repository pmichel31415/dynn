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
        shortcut_transform=None,
        layer_weight=1.0,
        shortcut_weight=1.0,
    ):
        super(ResidualLayer, self).__init__("residual")
        self.layer = layer
        self.shortcut_transform = _default_value(
            shortcut_transform, IdentityLayer()
        )
        self.layer_weight = layer_weight
        self.shortcut_weight = shortcut_weight

    def init(self, *args, **kwargs):
        """Initialize the layer before performing computation

        For example setup dropout, freeze some parameters, etc...
        """
        self.layer.init(*args, **kwargs)
        self.shortcut_transform.init(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        """Execute forward pass"""
        output = self.layer_weight * self.layer(*args, **kwargs)
        shortcut_connections = self.shortcut_weight * \
            self.shortcut_transform(*args, **kwargs)
        return output + shortcut_connections
