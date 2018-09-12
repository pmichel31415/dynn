#!/usr/bin/env python3

import numpy as np
import dynet as dy

OneInit = dy.ConstInitializer(1)
ZeroInit = dy.ConstInitializer(0)


def UniformInit(scale):
    return dy.UniformInitializer(scale)


def NormalInit(mean=0, std=1):
    return dy.NormalInitializer(mean, np.sqrt(std))
