"""
Data
====

This module contains helper functions and classes to manage data. This includes
code for minibatching as well as functions for downloading common datasets.

Supported datasets are:

- `MNIST <http://yann.lecun.com/exdb/mnist/>`_
"""
from . import (
    batching,
    mnist,
    cifar10,
)

__all__ = [
    "batching",
    "mnist",
    "cifar10",
]
