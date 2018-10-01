"""
Data
====

This module contains helper functions and classes to manage data. This includes
code for minibatching as well as functions for downloading common datasets.

Supported datasets are:

- `MNIST <http://yann.lecun.com/exdb/mnist/>`_
- `CIFAR-10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_
"""
from . import (
    batching,
    mnist,
    cifar10,
    dictionary,
    trees,
)

__all__ = [
    "batching",
    "mnist",
    "cifar10",
    "dictionary",
    "trees",
]
