#!/usr/bin/env python3
"""
Convolution layers
==================
"""
import dynet as dy

from ..parameter_initialization import ZeroInit
from ..activations import identity
from ..util import squeeze, unsqueeze, conditional_dropout
from .base_layers import ParametrizedLayer


class Conv1DLayer(ParametrizedLayer):
    """1D convolution along the first dimension

    Args:
        pc (:py:class:`dynet.ParameterCollection`): Parameter collection to
            hold the parameters
        input_dim (int): Input dimension
        num_kernels (int): Number of kernels (essentially the output dimension)
        kernel_width (int): Width of the kernels
        activation (function, optional): activation function
            (default: ``identity``)
        dropout (float, optional):  Dropout rate (default 0)
        nobias (bool, optional): Omit the bias (default ``False``)
    """

    def __init__(
        self,
        pc,
        input_dim,
        num_kernels,
        kernel_width,
        activation=identity,
        dropout_rate=0.0,
        nobias=False,
    ):
        super(Conv1DLayer, self).__init__(pc, "conv1d")
        # Hyper-parameters
        self.input_dim = input_dim
        self.num_kernels = num_kernels
        self.kernel_width = kernel_width
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.nobias = nobias
        # Parameters
        # Filters have shape:
        #   input_dim x kernel_width x 1 x num_filters
        self.K_p = self.pc.add_parameters(
            (1, self.kernel_width, self.input_dim, self.num_kernels), name="K"
        )
        if not self.nobias:
            self.b_p = self.pc.add_parameters(
                self.num_kernels, name="b", init=ZeroInit()
            )

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

    def __call__(self, x, stride=1, zero_padded=True):
        """Forward pass

        Args:
            x (:py:class:`dynet.Expression`): Input expression with the shape
                (length, input_dim)
            stride (int, optional): Stride along the temporal dimension
            zero_padded (bool, optional): Pad the image with zeros so that the
                output has the same length (default ``True``)

        Returns:
            :py:class:`dynet.Expression`: :math:`y=f(Wx+b)`
        """
        # Dropout
        x = conditional_dropout(x, self.dropout_rate, self.test)
        # Reshape the ``length x input_dim`` matrix to an
        # "image" of shape `` 1 x length x input_dim`` to use dynet's conv2d
        x = unsqueeze(x, 0)
        # Convolution
        is_valid = not zero_padded
        if self.nobias:
            output_img = dy.conv2d(
                x, self.K, stride=[1, stride], is_valid=is_valid
            )
        else:
            output_img = dy.conv2d_bias(
                x, self.K, self.b, stride=[1, stride], is_valid=is_valid
            )
        # Reshape back to a  ``length x output_dim`` matrix
        output = squeeze(output_img, 0)
        # Activation
        output = self.activation(output)
        # Final output
        return output


class Conv2DLayer(ParametrizedLayer):
    """2D convolution

    Args:
        pc (:py:class:`dynet.ParameterCollection`): Parameter collection to
            hold the parameters
        num_channels (int): Number of channels in the input image
        num_kernels (int): Number of kernels (essentially the output dimension)
        kernel_width (int): Width of the kernels
        kernel_height (int): Height of the kernels
        activation (function, optional): activation function
            (default: ``identity``)
        dropout (float, optional):  Dropout rate (default 0)
        nobias (bool, optional): Omit the bias (default ``False``)
    """

    def __init__(
        self,
        pc,
        num_channels,
        num_kernels,
        kernel_width,
        kernel_height,
        activation=identity,
        dropout_rate=0.0,
        nobias=False,
        zero_padded=True,
        strides=[1, 1]
    ):
        super(Conv2DLayer, self).__init__(pc, "conv2d")
        # Hyper-parameters
        self.num_channels = num_channels
        self.num_kernels = num_kernels
        self.kernel_width = kernel_width
        self.kernel_height = kernel_height
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.nobias = nobias
        # Parameters
        # Filters have shape:
        # kernel_height x kernel_width x num_channels x num_kernels
        self.K_p = self.pc.add_parameters(
            (
                self.kernel_height,
                self.kernel_width,
                self.num_channels,
                self.num_kernels
            ),
            name="K"
        )
        if not self.nobias:
            self.b_p = self.pc.add_parameters(
                self.num_kernels, name="b", init=ZeroInit()
            )

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

    def __call__(self, x, strides=1, zero_padded=True):
        """Forward pass

        Args:
            x (:py:class:`dynet.Expression`): Input expression with the shape
                (length, input_dim)
            zero_padded (bool, optional): Pad the image with zeros so that the
                output has the same width/height (default ``True``)
            strides (list, optional): Stride along width/height
                (list of size 2)

        Returns:
            :py:class:`dynet.Expression`: Convolved image
        """
        # Dropout
        x = conditional_dropout(x, self.dropout_rate, self.test)
        # Convolution
        is_valid = not zero_padded
        if self.nobias:
            output = dy.conv2d(
                x, self.K, stride=strides, is_valid=is_valid
            )
        else:
            output = dy.conv2d_bias(
                x, self.K, self.b, stride=strides, is_valid=is_valid
            )
        # Activation
        output = self.activation(output)
        # Final output
        return output
