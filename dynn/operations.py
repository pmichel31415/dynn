#!/usr/bin/env python3
"""
Operations
==========

This extends the base ``dynet`` library with useful operations.
"""

import dynet as dy


def squeeze(x, d=0):
    """Removes a dimension of size 1 at the given position

    Example:

    .. code-block: : python

        # (1, 20)
        x = dy.zeros((1, 20))
        # (20,)
        squeeze(x, 0)
        # (20, 1)
        x = dy.zeros((20, 1))
        # (20,)
        squeeze(x, 1)
        # (20,)
        squeeze(x, -1)
    """
    dim, batch_size = x.dim()
    if d < 0:
        d += len(dim)
    if d < 0 or d >= len(dim):
        raise ValueError(
            f"Dimension {d} out of bounds for {len(dim)}-dimensional "
            f"expression"
        )
    if dim[d] != 1:
        raise ValueError(
            f"Trying to squeeze dimension {d} of size {dim[d]}!=1"
        )
    new_dim = tuple(v for i, v in enumerate(dim) if i != d)
    return dy.reshape(x, new_dim, batch_size=batch_size)


def unsqueeze(x, d=0):
    """Insert a dimension of size 1 at the given position

    Example:

    .. code-block: : python

        # (10, 20)
        x = dy.zeros((10, 20))
        # (1, 10, 20)
        unsqueeze(x, 0)
        # (10, 20, 1)
        unsqueeze(x, -1)
    """
    dim, batch_size = x.dim()
    if d < 0:
        d += len(dim)+1
    if d < 0 or d > len(dim):
        raise ValueError(
            f"Cannot insert dimension at position {d} out of bounds "
            f"for {len(dim)}-dimensional expression"
        )
    new_dim = list(dim)
    new_dim.insert(d, 1)
    return dy.reshape(x, tuple(new_dim), batch_size=batch_size)
