#!/usr/bin/env python3
"""
Operations
==========

This extends the base ``dynet`` library with useful operations.
"""
from collections import Iterable

import numpy as np
import dynet as dy


def squeeze(x, d=0):
    """Removes a dimension of size 1 at the given position

    Example:

    .. code-block:: python

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

    .. code-block:: python

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


def stack(xs, d=0):
    """Like concatenated but inserts a dimension

    ``d=-1`` to insert a dimension at the last position

    Args:
        xs (list): List of expressions with the same dimensions
        d (int, optional): Position of the dimension ypu want to insert
    """
    xs = [unsqueeze(x, d=d) for x in xs]
    dim, _ = xs[0].dim()
    if d < 0:
        d += len(dim)
    if d < 0 or d > len(dim):
        raise ValueError(
            f"Cannot insert dimension at position {d} out of bounds "
            f"for {len(dim)}-dimensional expressions"
        )
    return dy.concatenate(xs, d=d)


def nll_softmax(logit, y):
    """This is the same as ``dy.pickneglogsoftmax``.

    The main difference is the shorter name and transparent handling of
    batches.
    It computes:

    .. math::

        -\\texttt{logit[y]}+\log(\sum_{\\texttt{c'}}e^{logit[c']})

    (softmax then negative log likelihood of ``y``)

    Args:
        logit (:py:class:`dynet.Expression`): Logit
        y (int,list): Either a class or a list of class
            (if ``logit`` is batched)
    """
    if isinstance(y, int) and logit.dim()[1] == 1:
        return dy.pickneglogsoftmax(logit, y)
    elif isinstance(y, Iterable) and logit.dim()[1] == len(y):
        return dy.pickneglogsoftmax_batch(logit, y)


def seq_mask(size, lengths, base_val=1, mask_val=0, left_aligned=True):
    """Returns a mask for a batch sequences of different lengths.

    This will return a ``(size,), len(lengths)`` shaped expression where the
    ``i`` th element of batch ``b`` is ``base_val`` iff ``i<=lengths[b]``
    (and ``mask_val`` otherwise).

    For example, if ``size`` is ``4`` and ``lengths`` is ``[1,2,4]`` then the
    returned mask will be:

    .. code-block::

        1 0 0 0
        1 1 0 0
        1 1 1 1

    (here each row is a batch element)

    Args:
        size (int): Max size of the sequence (must be ``>=max(lengths)``)
        lengths (list): List of lengths
        base_val (int, optional): Value of the mask for non-masked indices
            (typically 1 for multiplicative masks and 0 for additive masks).
            Defaults to 1.
        mask_val (int, optional): Value of the mask for masked indices
            (typically 0 for multiplicative masks and -inf for additive masks).
            Defaults to 0.
        left_aligned (bool, optional): Defaults to True.

    Returns:
        ::py:class:`dynet.Expression`: Mask expression
    """

    lengths = np.asarray(lengths)
    bsz = len(lengths)
    indices = np.stack([np.full(bsz, i) for i in range(size)], axis=0)
    # Which indices should be masked
    if left_aligned:
        should_mask = (indices >= lengths).astype(float)
    else:
        should_mask = (indices < (size - lengths)).astype(float)
    # Actual mask value
    mask_val = (1 - should_mask) * base_val + should_mask * mask_val
    # Return an expression
    return dy.inputTensor(mask_val, batched=True)
