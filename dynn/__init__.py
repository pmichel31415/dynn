"""
DyNN
====
"""
import numpy as np
import dynet as dy

from . import layers
from . import data
from . import operations
from . import activations
from . import parameter_initialization

__version__ = "0.0.12"

__all__ = [
    "layers",
    "data",
    "operations",
    "activations",
    "parameter_initialization"
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
