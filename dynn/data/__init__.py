"""
Data
====

This module contains helper functions and classes to manage data. This includes
code for minibatching as well as functions for downloading common datasets.

Supported datasets are:

- `MNIST <http://yann.lecun.com/exdb/mnist/>`_
- `CIFAR-10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_
- `SST <https://nlp.stanford.edu/sentiment/index.html>`_
- `PTB <http://www.fit.vutbr.cz/~imikolov/rnnlm>`_
"""
from . import (
    batching,
    preprocess,
    mnist,
    cifar10,
    sst,
    ptb,
    dictionary,
    trees,
)

__all__ = [
    "batching",
    "preprocess",
    "mnist",
    "cifar10",
    "sst",
    "ptb",
    "dictionary",
    "trees",
]
