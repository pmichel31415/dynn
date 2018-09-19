#!/usr/bin/env python3
"""
Pooling layers
==============
"""


from __future__ import print_function, division

import dynet as dy
from .. import util, operations
from .base_layers import BaseLayer


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


class MaxPooling1DLayer(BaseLayer):
    """1D max pooling

    Args:
        default_kernel_size (int, optional): Default kernel size. If this is
            not specified, the default is to pool over the full sequence
            (default: ``None``)
        stride (int, optional): Default temporal stride (default: ``1``)
    """

    def __init__(self, default_kernel_size=None, default_stride=1):
        super(MaxPooling1DLayer, self).__init__("maxpool1d")
        self.kernel_size = default_kernel_size
        self.stride = default_stride

    def __call__(self, x, kernel_size=None, stride=None):
        """Max pooling over the first dimension.

        This takes either a list of ``N`` ``d``-dimensional vectors or
        a ``N x d`` matrix.

        The output will be a matrix of dimension
        ``(N - kernel_size + 1) // stride x d``

        Args:
            x (:py:class:`dynet.Expression`): Input matrix or list of vectors
            dim (int, optional): The reduction dimension (default: ``0``)
            kernel_size (int, optional): Kernel size. If this is not
                specified, the default size specified in the constructor
                is used.
            stride (int, optional): Temporal stride. If this is not specified,
                the default stride specified in the constructor is used.

        Returns:
            :py:class:`dynet.Expression`: Pooled sequence.
        """
        # Convert to matrix if needed
        x = util.list_to_matrix(x)
        # x's dimension,x_dim = length x dimension
        x_dim, _ = x.dim()
        # Reshape as length x 1 x dimension "image" to use maxpooling2d
        img = operations.unsqueeze(x, d=1)
        # If the kernel size is None, set it to the length of the sentence
        kernel_size = [kernel_size or self.kernel_size or x_dim[0], 1]
        # 2D pooling with appropriate kernel size
        max_pooled_img = dy.maxpooling2d(
            img,
            ksize=kernel_size,
            stride=[stride or self.stride, 1],
            is_valid=True,
        )
        # Squeeze the useless dimension to get a matrix
        output = operations.squeeze(max_pooled_img, 1)
        # Final output
        return output


class MaxPooling2DLayer(BaseLayer):
    """2D max pooling.

    Args:
        kernel_size (list, optional): Default kernel size. This is a list of
            two elements, one per dimension. If either is not specified, the
            default is to pool over the entire dimension
            (default: ``[None, None]``)
        default_strides (list, optional): Stride along each dimension
            (list of size 2, defaults to ``[1, 1]``).
    """

    def __init__(self, default_kernel_size=None, default_strides=None):
        super(MaxPooling2DLayer, self).__init__("maxpool1d")
        self.kernel_size = util._default_value(
            default_kernel_size, [None, None]
        )
        self.strides = util._default_value(default_strides, [1, 1])

    def __call__(self, x, kernel_size=None, strides=None):
        """Max pooling over the first dimension.

        If either of the ``kernel_size`` elements is not specified, the
        pooling will be done over the full dimension (and the stride is
        ignored)

        Args:
            x (:py:class:`dynet.Expression`): Input image (3-d tensor) or
                matrix.
            kernel_size (list, optional): Size of the pooling kernel. If this
                is not specified, the default specified in the constructor is
                used.
            strides (list, optional): Stride along width/height. If this is not
                specified, the default specified in the constructor is used.

        Returns:
            :py:class:`dynet.Expression`: Pooled sequence.
        """
        # Convert to matrix if needed
        x = util.list_to_matrix(x)
        # x's dimension
        x_dim, _ = x.dim()
        # If there is no channel dimension, add it
        if len(x_dim) < 3:
            x = operations.unsqueeze(x, d=-1)
        # If the kernel size is None, set it to the size of the dimension
        kernel_size = util._default_value(kernel_size, [None, None])
        kernel_size = [
            kernel_size[0] or self.kernel_size[0] or x_dim[0],
            kernel_size[1] or self.kernel_size[1] or x_dim[1]
        ]
        # Strides
        strides = util._default_value(strides, [None, None])
        strides = [
            strides[0] or self.strides[0] or 1,
            strides[1] or self.strides[1] or 1
        ]
        # 2D pooling with appropriate kernel size
        max_pooled_img = dy.maxpooling2d(
            x, ksize=kernel_size, stride=strides, is_valid=True,
        )
        # If there was no 3rd dimension in the input, remove it from the output
        if len(x_dim) < 3:
            max_pooled_img = operations.squeeze(max_pooled_img, d=-1)
        # Final output
        return max_pooled_img
