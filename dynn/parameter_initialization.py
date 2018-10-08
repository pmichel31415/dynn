#!/usr/bin/env python3
"""
Parameter initialization
========================

Some of those are just less verbose versions of
dynet's ``PyInitializer`` s
"""
import dynet as dy


def OneInit():
    """Initialize with :math:`1`

    Returns:
        :py:class:`dynet.PyInitializer`: ``dy.ConstInitializer(1)``
    """

    return dy.ConstInitializer(1)


def ZeroInit():
    """Initialize with :math:`0`

    Returns:
        :py:class:`dynet.PyInitializer`: dy.ConstInitializer(0)
    """
    return dy.ConstInitializer(0)


def UniformInit(scale=1.0):
    """Uniform initialization between ``-scale`` and ``scale``

    Args:
        scale (float): Scale of the distribution

    Returns:
        :py:class:`dynet.PyInitializer`:
            ``dy.UniformInitializer(scale)``
    """

    return dy.UniformInitializer(scale)


def NormalInit(mean=0, std=1):
    """Gaussian initialization

    Args:
        mean (int, optional): Mean (default: 0.0)
        std (int, optional): Standard deviation (:math:`\\neq` variance)
            (default: 1.0)

    Returns:
        :py:class:`dynet.PyInitializer`:
            ``dy.NormalInitializer(mean, sqrt(std))``
    """

    return dy.NormalInitializer(mean, std**2)
