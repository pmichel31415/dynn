#!/usr/bin/env python3
"""
Training helper functions and classes
=====================================

Adds new optimizers and LR schedules to dynet.
"""
import numpy as np


def inverse_sqrt_schedule(warmup, lr0):
    """Inverse square root learning rate schedule

    At step :math:`t` , the learning rate has value

    .. math::

        \\texttt{lr}\\times
        \min(1 {\sqrt{t}},
        \sqrt{\\frac {t} {\\texttt{warmup}^3})

    Args:
        warmup (int): Number of warmup steps
        lr0 (float): Initial learning rate
    """

    step = 0
    while True:
        scale = min(1/np.sqrt(step+1e-20), step/np.sqrt(warmup**3))
        step += 1
        yield lr0 * scale
