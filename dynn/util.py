#!/usr/bin/env python3
"""Utility functions"""

import dynet as dy


def list_to_matrix(l):
    """Transforms a list of N vectors of dimension d into a (d, N) matrix"""
    if isinstance(l, list):
        return dy.concatenate_cols(l)
    else:
        return l


def matrix_to_image(M):
    """Transforms a matrix (d1, d2) into an 'image' with one channel
    (d1, d2, 1)"""
    dim, bsize = M.dim()
    if len(dim) == 1:
        return dy.reshape(M, (dim[0], 1, 1), batch_size=bsize)
    elif len(dim) == 2:
        d1, d2 = dim
        return dy.reshape(M, (d1, d2, 1), batch_size=bsize)
    elif len(dim) == 3:
        return M
    else:
        raise ValueError('Cannot convert tensor of order >3 to image')


def image_to_matrix(M):
    """Transforms an 'image' with one channel (d1, d2, 1) into a matrix
    (d1, d2)"""
    dim, bsize = M.dim()
    if len(dim) == 3:
        d1, d2, d3 = dim
        assert d3 == 1, 'Input image has more than 1 channel'
        return dy.reshape(M, (d1, d2), batch_size=bsize)
    else:
        return M


def conditional_dropout(x, dropout_rate, flag):
    """This helper function applies dropout only if the flag
    is set to ``True`` and the ``dropout_rate`` is positive.

    Args:
        x (dynet.Expression): Input expression
        dropout_rate (float): Dropout rate
        flag (bool): Setting this to false ensures that dropout
        is never applied (for testing for example)
    """
    if dropout_rate > 0 and flag:
        return dy.dropout(x, dropout_rate)
    else:
        return x
