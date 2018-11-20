"""
DyNN
====
"""
import numpy as np
import dynet as dy

from . import command_line
from . import layers
from . import data
from . import io
from . import operations
from . import activations
from . import parameter_initialization
from . import training

__version__ = "0.1"

__all__ = [
    "command_line",
    "layers",
    "data",
    "io",
    "operations",
    "activations",
    "parameter_initialization",
    "training"
]


def set_random_seed(seed, numpy_seed=None):
    """Set the random seeds.

    This will fix the random seeds both for dynet (influences parameter
    initialization, stochastic operations etc...) and numpy (used for data
    shuffling mostly).

    Args:
        seed (int): Random seed
        numpy_seed (int, optional): Optionally you can specify a different seed
            for numpy (by default it will seed dynet with ``seed`` and numpy
            with ``seed+1``)
    """
    numpy_seed = numpy_seed or (seed + 1)
    dy.reset_random_seed(seed)
    np.random.seed(numpy_seed)
