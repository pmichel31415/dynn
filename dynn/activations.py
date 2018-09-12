#!/usr/bin/env python3
"""
Activation functions
====================

Common activation functions for neural networks.

Most of those are wrappers around standard dynet operations
(eg. relu == rectify)
"""

import dynet as dy


def identity(x):
    """The identity function"""
    return x


def tanh(x):
    """The hyperbolic tangent function"""
    return dy.tanh(x)


def sigmoid(x):
    """The sigmoid function"""
    return dy.logistic(x)


def relu(x):
    """The REctified Linear Unit"""
    return dy.rectify(x)
