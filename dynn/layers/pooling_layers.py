# -*- coding: utf-8 -*-
"""Pooling layers"""


from __future__ import print_function, division

import dynet as dy
from layers import Layer
import util


def max_pool_dim(x, d=0, kernel_width=None, stride=1):
    """Efficent max pooling on GPU, assuming x is a matrix
    or a list of vectors"""
    # this is a hack to use the cudnn maxpooling_2d
    # until dynet's max_dim gets faster

    # If x is a list of d2 elements of size d1,
    # concatenate it to a (d1, d2) matrix
    h = util.list_to_matrix(x)
    # Reshape as (d1, d2, 1) "image" if necessary
    h = util.matrix_to_image(h)
    # Retrieve shape
    (d1, d2, _), bsize = h.dim()
    # Kernel size
    kernel_size = [d1, d2]
    kernel_size[1-d] = 1
    if kernel_width is not None:
        kernel_size[d] = kernel_width
    # 2D pooling with convenient kernel size
    max_pooled = dy.maxpooling2d(h, ksize=kernel_size, stride=[1, stride])
    # The output has shape (1,d2,1) or (d1,1,1), needs reshaping
    output_dim = d1 if d == 0 else d2
    output = dy.reshape(max_pooled, (output_dim,), batch_size=bsize)
    return output


class MaxPooling1DLayer(Layer):
    """1D max pooling

    Args:
        pc (dynet.ParameterCollection): Parameter collection to
        hold the parameters
        kernel_width (int, optional): Kernel width. If this is not specified,
            the default is to pool over the full sequence (default: ``None``)
        stride (int, optional): Temporal stride (default: ``1``)
    """

    def __init__(self, pc, kernel_width=None, stride=1):
        super(MaxPooling1DLayer, self).__init__(pc, 'maxpool1d')
        # Hyper-parameter
        self.kernel_width = kernel_width
        self.stride = stride

    def init(self, test=False, update=True):
        pass

    def __call__(self, x):
        """Forward pass

        Args:
            x (dynet.Expression): Input expression with the shape
                (length, input_dim)

        Returns:
            dynet.Expression: Vector of size input_dim
        """
        # Max-pooling
        self.h = max_pool_dim(
            x, d=1, kernel_width=self.kernel_width, stride=self.stride
        )
        # Final output
        return self.h
