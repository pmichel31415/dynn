#!/usr/bin/env python3
"""This contains initializers

Some of those are just less verbose versions of
dynet's ``ParameterInitializer`` s
"""
import numpy as np
import dynet as dy

OneInit = dy.ConstInitializer(1)
ZeroInit = dy.ConstInitializer(0)


def UniformInit(scale):
    return dy.UniformInitializer(scale)


def NormalInit(mean=0, std=1):
    return dy.NormalInitializer(mean, np.sqrt(std))
