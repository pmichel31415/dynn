#!/usr/bin/env python3
"""
Residual layers
===============
"""


from .base_layers import BaseLayer
from .functional_layers import IdentityLayer


class Residual(BaseLayer):
    """Adds residual connections to a layer"""

    def __init__(
        self,
        layer,
        shortcut_transform=None,
        layer_weight=1.0,
        shortcut_weight=1.0,
    ):
        super(Residual, self).__init__("residual")
        self.layer = layer
        if shortcut_transform is None:
            self.shortcut_transform = IdentityLayer()
        else:
            self.shortcut_transform = shortcut_transform
        self.layer_weight = layer_weight
        self.shortcut_weight = shortcut_weight

    def __call__(self, *args, **kwargs):
        """Execute forward pass"""
        output = self.layer_weight * self.layer(*args, **kwargs)
        shortcut_connections = self.shortcut_weight * \
            self.shortcut_transform(*args, **kwargs)
        return output + shortcut_connections
