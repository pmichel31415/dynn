#!/usr/bin/env python3


class BaseLayer(object):
    """Base layer class
    Layers are typically used like this:

    Example:

        .. code-block:: python

            # Instantiate layer
            layer = Layer(parameter_collection, *args, **kwargs)
            # [...]
            # Renew computation graph
            dy.renew_cg()
            # Initialize layer
            layer.init(*args, **kwargs)
            # Apply layer forward pass
            y = layer(x)
    """

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
