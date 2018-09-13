#!/usr/bin/env python3
"""
Base layer
==========
"""


class BaseLayer(object):
    """Base layer interface"""

    def init(self, *args, **kwargs):
        """Initialize the layer before performing computation

        For example setup dropout, freeze some parameters, etc...
        """
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        """Execute forward pass"""
        raise NotImplementedError()


class ParametrizedLayer(BaseLayer):
    """This is the base class for layers with trainable parameters"""

    def __init__(self, pc, name):
        """Creates a subcollection for this layer with a custom name"""
        self.pc = pc.add_subcollection(name=name)
