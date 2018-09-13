#!/usr/bin/env python3
"""
Convolution layers
==================
"""
import dynet as dy

from ..parameter_initialization import ZeroInit
from ..activations import identity
from ..util import matrix_to_image, image_to_matrix
from .base_layers import ParametrizedLayer


class Conv1DLayer(ParametrizedLayer):
    """1D convolution

    Args:
        input_dim (int): Input dimension
        num_kernels (int): Number of kernels (essentially the output dimension)
        kernel_width (int): Width of the kernels
        pc (:py:class:`dynet.ParameterCollection`): Parameter collection to
            hold the parameters
        activation (function, optional): activation function
            (default: ``identity``)
        dropout (float, optional):  Dropout rate (default 0)
        nobias (bool, optional): Omit the bias (default ``False``)
        strides (int, optional): Stride along the temporal dimension
    """

    def __init__(
        self,
        input_dim,
        num_kernels,
        kernel_width,
        pc,
        activation=identity,
        dropout=0.0,
        nobias=False,
        strides=1
    ):
        super(Conv1DLayer, self).__init__(pc, 'conv1d')
        # Hyper-parameters
        self.input_dim = input_dim
        self.num_kernels = num_kernels
        self.kernel_width = kernel_width
        self.strides = strides
        self.nobias = nobias
        # Filters have shape:
        #   input_dim x kernel_width x 1 x num_filters
        self.K_p = self.pc.add_parameters(
            (self.di, self.kw, 1, self.nk), name='K')
        if not self.nobias:
            self.b_p = self.pc.add_parameters(
                self.nk, name='b', init=ZeroInit())
        # Dropout
        self.dropout = dropout
        # Activation function
        self.activation = activation

    def init(self, test=False, update=True):
        """Initialize the layer before performing computation

        Args:
            test (bool, optional): If test mode is set to ``True``,
                dropout is not applied (default: ``True``)
            update (bool, optional): Whether to update the parameters
                (default: ``True``)
        """
        # Initialize parameters
        self.K = self.K_p.expr(update)
        if not self.nobias:
            self.b = self.b_p.expr(update)

        self.test = test

    def __call__(self, x):
        """Forward pass

        Args:
            x (:py:class:`dynet.Expression`): Input expression with the shape
                (length, input_dim)

        Returns:
            :py:class:`dynet.Expression`: :math:`y=f(Wx+b)`
        """
        # Dropout
        if not self.test and self.dropout > 0:
            x = dy.dropout(x, self.dropout)
        # Reshape the ``length x input_dim`` matrix to an
        # "image" to use dynet's conv2d
        x = matrix_to_image(x)
        # Convolution
        if self.nobias:
            self.a = dy.conv2d(
                x, self.K, stride=[1, self.strides], is_valid=False
            )
        else:
            self.a = dy.conv2d_bias(
                x, self.K, self.b, stride=[1, self.strides], is_valid=False
            )
        # Reshape back to a  ``length x output_dim`` matrix
        self.a = image_to_matrix(self.a)
        # Activation
        self.h = self.activation(self.a)
        # Final output
        return self.h
