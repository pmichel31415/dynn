# -*- coding: utf-8 -*-
"""Pooling layers
"""


from __future__ import print_function, division

import sys
import dynet as dy

def max_pool_1d(x, d=0):
    """Efficent max pooling on GPU, assuming x is a matrix or a list of vectors"""
    # If x is a list of d2 elements of size d1, concatenate it to a (d1, d2) matrix
    if isinstance(x, list):
        x = dy.concatenate_cols(x)
    # Retrieve shape
    (d1, d2), bsize = x.dim()
    # Add the extra dimension (channel) TODO: verify that this is necessary
    h = dy.reshape(x, (d1, d2, 1), batch_size=bsize)
    # Kernel size
    kernel_size = [d1, d2]
    kernel_size[1-d] = 1
    # 2D pooling with convenient kernel size
    max_pooled = dy.maxpooling2d(h, ksize=kernel_size, stride=[1, 1])
    # The output has shape (1,d2,1) or (d1,1,1), needs reshaping
    output_dim = d1 if d==0 else d2
    output = dy.reshape(max_pooled, (output_dim,), batch_size=bsize)
    return output