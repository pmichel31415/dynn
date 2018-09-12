#!/usr/bin/env python3
"""Standard densely connected layers"""
import dynet as dy

from ..parameter_initialization import ZeroInit
from .base_layer import BaseLayer


class DenseLayer(BaseLayer):
    """Densely connected layer

    :math:`y=f(Wx+b)`

    Args:
        pc (dynet.ParameterCollection): Parameter collection to
            hold the parameters
        input_dim (int): Input dimension
        output_dim (int): Output dimension
        activation (function, optional): activation function
            (default: ``dynet.tanh``)
        dropout (float, optional):  Dropout rate (default 0)
        nobias (bool, optional): Omit the bias (default ``False``)
    """

    def __init__(
        self,
        pc,
        input_dim,
        output_dim,
        activation=dy.tanh,
        dropout=0.0,
        nobias=False,
    ):
        super(DenseLayer, self).__init__(pc, "dense")
        self.W_p = self.pc.add_parameters((output_dim, input_dim), name="W")
        if not nobias:
            self.b_p = self.pc.add_parameters(
                output_dim, name="b", init=ZeroInit)

        self.dropout = dropout
        self.nobias = nobias
        self.activation = activation

    def init(self, test=False, update=True):
        """Initialize the layer before performing computation

        Args:
            test (bool, optional): If test mode is set to ``True``,
                dropout is not applied (default: ``True``)
            update (bool, optional): Whether to update the parameters
                (default: ``True``)
        """

        self.W = self.W_p.expr(update)
        if not self.nobias:
            self.b = self.b_p.expr(update)

        self.test = test

    def __call__(self, x):
        """Forward pass

        Args:
            x (dynet.Expression): Input expression (a vector)

        Returns:
            dynet.Expression: :math:`y=f(Wx+b)`
        """
        # Dropout
        if not self.test and self.dropout > 0:
            x = dy.dropout(x, self.dropout)
        # Output
        if self.nobias:
            self.h = self.W * x
        else:
            self.h = self.activation(dy.affine_transform([self.b, self.W, x]))

        # Final output
        return self.h


class GatedLayer(BaseLayer):
    """Gated linear layer:

    :math:`y=(W_ox+b_o)\circ \sigma(W_gx+b_g)`

    Args:
        pc (dynet.ParameterCollection): Parameter collection to
            hold the parameters
        input_dim (int): Input dimension
        output_dim (int): Output dimension
        activation (function, optional): activation function
            (default: ``dynet.tanh``)
        dropout (float, optional):  Dropout rate (default 0)
    """

    def __init__(
        self,
        pc,
        input_dim,
        output_dim,
        activation=dy.tanh,
        dropout=0.0,
    ):
        super(GatedLayer, self).__init__(pc, "gated")
        # Hyperparameters
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.activation = activation
        # Affine layer parameters
        self.Wo_p = self.pc.add_parameters((output_dim, input_dim), name="Wo")
        self.bo_p = self.pc.add_parameters(
            output_dim, name="bo", init=ZeroInit
        )
        # Gating layer parameters
        self.Wg_p = self.pc.add_parameters((output_dim, input_dim), name="Wg")
        self.bg_p = self.pc.add_parameters(
            output_dim, name="bg", init=ZeroInit
        )

    def init(self, test=False, update=True):
        """Initialize the layer before performing computation

        Args:
            test (bool, optional): If test mode is set to ``True``,
                dropout is not applied (default: ``True``)
            update (bool, optional): Whether to update the parameters
                (default: ``True``)
        """
        self.Wo = self.Wo_p.expr(update)
        self.bo = self.bo_p.expr(update)

        self.Wg = self.Wg_p.expr(update)
        self.bg = self.bg_p.expr(update)

        self.test = test

    def __call__(self, x):
        """Forward pass

        Args:
            x (dynet.Expression): Input expression (a vector)

        Returns:
            dynet.Expression: :math:`y=(W_ox+b_o)\circ \sigma(W_gx+b_g)`
        """

        # Output
        self.o = self.activation(dy.affine_transform([self.bo, self.Wo, x]))
        # Gate
        self.g = dy.logistic(dy.affine_transform([self.bg, self.Wg, x]))
        # final output
        return dy.cmult(self.g, self.o)


class StackedLayers(BaseLayer):
    """Helper class to stack layers"""

    def __init__(self, *args):
        self.layers = args

    def init(self, test=False, update=True):
        for layer in self.layers:
            layer.init(test=test, update=update)

    def __call__(self, x):
        self.hs = [x]
        for layer in self.layers:
            self.hs.append(layer(self.hs[-1]))
        return self.hs[-1]
