#!/usr/bin/env python3
"""
Densely connected layers
========================
"""
import dynet as dy

from ..activations import identity
from ..parameter_initialization import ZeroInit
from ..util import conditional_dropout
from .base_layers import ParametrizedLayer


class Affine(ParametrizedLayer):
    """Densely connected layer

    :math:`y=f(Wx+b)`

    Args:
        pc (:py:class:`dynet.ParameterCollection`): Parameter collection to
            hold the parameters
        input_dim (int): Input dimension
        output_dim (int): Output dimension
        activation (function, optional): activation function
            (default: :py:function:`identity`)
        dropout (float, optional):  Dropout rate (default 0)
        nobias (bool, optional): Omit the bias (default ``False``)
    """

    def __init__(
        self,
        pc,
        input_dim,
        output_dim,
        activation=identity,
        dropout=0.0,
        nobias=False,
        W=None,
        b=None,
    ):
        super(Affine, self).__init__(pc, "affine")
        self.add_parameters("W", (output_dim, input_dim), param=W)
        if not nobias:
            self.add_parameters("b", output_dim, param=b, init=ZeroInit())

        self.dropout = dropout
        self.nobias = nobias
        self.activation = activation

    def __call__(self, x):
        """Forward pass.

        Args:
            x (:py:class:`dynet.Expression`): Input expression (a vector)

        Returns:
            :py:class:`dynet.Expression`: :math:`y=f(Wx+b)`
        """
        # Dropout
        x = conditional_dropout(x, self.dropout, not self.test)
        # Output
        if self.nobias:
            h = self.W * x
        else:
            h = dy.affine_transform([self.b, self.W, x])
        # Final output
        return self.activation(h)


class GatedLayer(ParametrizedLayer):
    """Gated linear layer:

    :math:`y=(W_ox+b_o)\circ \sigma(W_gx+b_g)`

    Args:
        pc (:py:class:`dynet.ParameterCollection`): Parameter collection to
            hold the parameters
        input_dim (int): Input dimension
        output_dim (int): Output dimension
        activation (function, optional): activation function
            (default: :py:class:`dynet.tanh`)
        dropout (float, optional):  Dropout rate (default 0)
    """

    def __init__(
        self,
        pc,
        input_dim,
        output_dim,
        activation=dy.tanh,
        dropout=0.0,
        Wo=None,
        bo=None,
        Wg=None,
        bg=None,
    ):
        super(GatedLayer, self).__init__(pc, "gated")
        # Hyperparameters
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.activation = activation
        # Affine layer parameters
        self.add_parameters("Wo", (output_dim, input_dim), param=Wo)
        self.add_parameters("bo", output_dim, param=bo, init=ZeroInit())
        # Gating layer parameters
        self.add_parameters("Wg", (output_dim, input_dim), param=Wg)
        self.add_parameters("bg", output_dim, param=bg, init=ZeroInit())

    def __call__(self, x):
        """Forward pass

        Args:
            x (:py:class:`dynet.Expression`): Input expression (a vector)

        Returns:
            :py:class:`dynet.Expression`:
                :math:`y=(W_ox+b_o)\circ \sigma(W_gx+b_g)`
        """

        # Output
        o = self.activation(dy.affine_transform([self.bo, self.Wo, x]))
        # Gate
        g = dy.logistic(dy.affine_transform([self.bg, self.Wg, x]))
        # final output
        return dy.cmult(g, o)
