"""
Layers
======

Layers are the standard unit of neural models in DyNN. Layers are typically
used like this:

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
from . import (
    base_layers,
    dense_layers,
    recurrent_layers,
    transduction_layers,
    pooling_layers,
    convolution_layers,
    normalization_layers,
)

__all__ = [
    "base_layers",
    "dense_layers",
    "recurrent_layers",
    "transduction_layers",
    "pooling_layers",
    "convolution_layers",
    "normalization_layers",
]
