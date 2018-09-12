#!/usr/bin/env python3
"""
Base layer
==========
"""


class BaseLayer(object):
    """Base layer interface"""

    def __init__(self, pc, name):
        """Creates a subcollection for this layer with a custom name"""
        self.pc = pc.add_subcollection(name=name)

    def init(self, *args, **kwargs):
        """Initialize the layer before performing computation

        For example setup dropout, freeze some parameters, etc...
        """
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        """Execute forward pass"""
        raise NotImplementedError()
