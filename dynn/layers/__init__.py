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
from .base_layers import BaseLayer, ParametrizedLayer

from .functional_layers import Lambda
from .dense_layers import Affine
from .embedding_layers import Embeddings
from .residual_layers import Residual
from .recurrent_layers import (
    RecurrentCell, StackedRecurrentCells, ElmanRNN, LSTM, StackedLSTM
)
from .transduction_layers import Transduction, Unidirectional, Bidirectional
from .pooling_layers import MaxPool1D, MaxPool2D, MeanPool1D
from .convolution_layers import Conv1D, Conv2D
from .flow_layers import Flatten
from .normalization_layers import LayerNormalization
from .combination_layers import Sequential, Parallel

__all__ = [
    "BaseLayer",
    "ParametrizedLayer",
    "Lambda",
    "Affine",
    "Embeddings",
    "Residual",
    "RecurrentCell",
    "StackedRecurrentCells",
    "ElmanRNN",
    "LSTM",
    "StackedLSTM",
    "Transduction",
    "Unidirectional",
    "Bidirectional",
    "MaxPool1D",
    "MaxPool2D",
    "MeanPool1D",
    "Conv1D",
    "Conv2D",
    "Flatten",
    "LayerNormalization",
    "Sequential",
    "Parallel",
]
