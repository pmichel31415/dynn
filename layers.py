# -*- coding: utf-8 -*-
"""Standard feed-foward layers

Layers are typically used like this:

Example:

    .. code-block:: python

        # Instantiate layer
        layer = Layer(parameter_collection, *args, **kwargs)
        # [...]
        # Renew computation graph
        dy.renew_cg()
        # Initialize layer
        layer.init(*args, **kwargs)
        # Apply layer forward pass
        y = layer(x)
"""


from __future__ import print_function, division

import sys
import dynet as dy

OneInit = dy.ConstInitializer(1)
ZeroInit = dy.ConstInitializer(0)

class Layer(object):
    """Base layer object"""
    def __init__(self, pc, name):
        """Creates a subcollection for this layer with a custom name"""
        self.pc = pc.add_subcollection(name=name)

    def init(self, *args, **kwargs):
        """Loads parameters into computation graph"""
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        """Execute forward pass"""
        raise NotImplementedError()


class DenseLayer(Layer):
    """Densely connected layer"""
    def __init__(self, di, dh, pc,activation=dy.tanh, dropout=0.0, nobias=False):
        super(DenseLayer, self).__init__(pc, 'dense')
        self.W_p = self.pc.add_parameters((dh, di), name='W')
        if not nobias:
            self.b_p = self.pc.add_parameters(dh, name='b', init=ZeroInit)

        self.dropout = dropout
        self.nobias = nobias
        self.activation = activation

    def init(self, test=False, update=True):
        self.W = self.W_p.expr(update)
        if not self.nobias:
            self.b = self.b_p.expr(update)

        self.test = test

    def __call__(self, x):
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

class GatedLayer(Layer):
    """Gated linear layer: :math:`y=(W_ox+b_o)\circ \sigma(W_gx+b_g)`"""
    def __init__(self, di, dh, pc,activation=dy.tanh, dropout=0.0):
        super(GatedLayer, self).__init__(pc, 'gated')
        self.Wo_p = self.pc.add_parameters((dh, di), name='Wo')
        self.bo_p = self.pc.add_parameters(dh, name='bo', init=ZeroInit)

        self.Wg_p = self.pc.add_parameters((dh, di), name='Wg')
        self.bg_p = self.pc.add_parameters(dh, name='bg', init=ZeroInit)

        self.activation = activation

    def init(self, test=False, update=True):
        self.Wo = self.Wo_p.expr(update)
        self.bo = self.bo_p.expr(update)

        self.Wg = self.Wg_p.expr(update)
        self.bg = self.bg_p.expr(update)

        self.test = test

    def __call__(self, x):
        # Output
        self.o = self.activation(dy.affine_transform([self.bo, self.Wo, x]))
        # Gate
        self.g = dy.logistic(dy.affine_transform([self.bg, self.Wg, x]))
        # final output
        return dy.cmult(self.g, self.o)

class LayerNormalization(Layer):
    """Layer normalization layer"""
    def __init__(self, di, pc,):
        super(LayerNormalization, self).__init__(pc, 'layer-norm')
        self.gain_p = self.pc.add_parameters(di, name='gain', init=OneInit)
        self.bias_p = self.pc.add_parameters(di, name='bias', init=ZeroInit)

    def init(self, test=False, update=True):
        self.gain = self.gain_p.expr(update)
        self.bias = self.bias_p.expr(update)

        self.test = test

    def __call__(self, x):
        # Output
        self.o = dy.layer_norm(x, self.gain, self.bias)
        # final output
        return self.o
