#!/usr/bin/env python3
"""
Activation functions
====================

Common activation functions for neural networks.

Most of those are wrappers around standard dynet operations
(eg. ``rectify`` -> ``relu``)
"""

import dynet as dy


def identity(x):
    """The identity function

    :math:`y=x`

    Args:
        x (:py:class:`dynet.Expression`): Input expression

    Returns:
        :py:class:`dynet.Expression`: :math:`x`
    """
    return x


def tanh(x):
    """The hyperbolic tangent function

    :math:`y=\\tanh(x)`

    Args:
        x (:py:class:`dynet.Expression`): Input expression

    Returns:
        :py:class:`dynet.Expression`: :math:`\\tanh(x)`
    """
    return dy.tanh(x)


def sigmoid(x):
    """The sigmoid function

    :math:`y=\\frac{1}{1+e^{-x}}`

    Args:
        x (:py:class:`dynet.Expression`): Input expression

    Returns:
        :py:class:`dynet.Expression`: :math:`\\frac{1}{1+e^{-x}}`
    """
    return dy.logistic(x)


def relu(x):
    """The REctified Linear Unit

    :math:`y=\max(0,x)`

    Args:
        x (:py:class:`dynet.Expression`): Input expression

    Returns:
        :py:class:`dynet.Expression`: :math:`\max(0,x)`
    """

    return dy.rectify(x)
