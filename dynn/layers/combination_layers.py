#!/usr/bin/env python3
"""
Combination layers
==================

Perhaps unsurprisingly, combination layers are layers that combine other
layers within one layer.
"""
import dynet as dy

from .. import util, operations
from .base_layers import BaseLayer


class Sequential(BaseLayer):
    """A helper class to stack layers into deep networks.

    Args:
        layers (list): A list of :py:class:`dynn.layers.BaseLayer`
            objects. The first layer is the first one applied to the input.
            It is the programmer's responsibility to make sure that the
            layers are compatible (eg. the output of each layer can be
            fed into the next one)
        default_return_last_only (bool, optional): Return only the output
        of the last layer (as opposed to the output of all layers).
    """

    def __init__(self, *layers, default_return_last_only=True):
        # Check number of arguments
        if len(layers) == 0:
            raise ValueError("Can't stack empty list of layers")
        # Check argument type
        for layer_idx, layer in enumerate(layers):
            if not isinstance(layer, BaseLayer):
                raise ValueError(
                    f"Layer #{layer_idx} should be a subclass of BaseLayer, "
                    f"is {layer.__class__}"
                )
        self.return_last_only = default_return_last_only
        self.layers = layers

    def init(self, test=False, update=True):
        for layer in self.layers:
            layer.init(test=test, update=update)

    def __call__(self, x, return_last_only=None):
        """Calls all the layers in succession.

        Computes ``layers[n-1](layers[n-2](...layers[0](x)))``

        Args:
            x (:py:class:`dynet.Expression`): Input expression
            return_last_only (bool, optional): Overrides the default

        Returns:
            :py:class:`dynet.Expression`, list: Depending on
                ``return_last_only``, returns either the last expression or a
                list of all the layer's outputs (first to last)
        """

        hs = [x]
        for layer in self.layers:
            hs.append(layer(hs[-1]))
        # Either return the last output or all outputs
        if return_last_only is None:
            return_last_only = self.return_last_only
        if return_last_only:
            return hs[-1]
        else:
            return hs[1:]


class Parallel(BaseLayer):
    """A helper class to run layers on the same input and concatenate their outputs

    This can be used to create 2d conv layers with multiple kernel sizes by
    concatenating multiple :py:class:`dynn.layers.Conv2D` .


    Args:
        layers (list): A list of :py:class:`dynn.layers.BaseLayer`
            objects. The first layer is the first one applied to the input.
            It is the programmer's responsibility to make sure that the
            layers are compatible (eg. each layer takes the same input and the
            outputs have the same shape everywhere except along the
            concatenation dimension)
        dim (int): The concatenation dimension
        default_insert_dim (bool, optional): Instead of concatenating along an
            existing dimension, insert a a new dimension at ``dim`` and
            concatenate.
    """

    def __init__(self, *layers, dim=0, default_insert_dim=False):
        # Check number of arguments
        if len(layers) == 0:
            raise ValueError("Can't concatenate empty list of layers")
        # Check argument type
        for layer_idx, layer in enumerate(layers):
            if not isinstance(layer, BaseLayer):
                raise ValueError(
                    f"Layer #{layer_idx} should be a subclass of BaseLayer, "
                    f"is {layer.__class__}"
                )

        self.layers = layers
        self.dim = dim
        self.insert_dim = default_insert_dim

    def init(self, test=False, update=True):
        for layer in self.layers:
            layer.init(test=test, update=update)

    def __call__(self, x, insert_dim=None):
        """Calls all the layers in succession.

        Computes ``dy.concatenate([layers[0](x) ...layers[n-1](x)], d=dim)``

        Args:
            x (:py:class:`dynet.Expression`): Input expression
            default_insert_dim (bool, optional): Override the default

        Returns:
            :py:class:`dynet.Expression`, list: Depending on
                ``return_last_only``, returns either the last expression or a
                list of all the layer's outputs (first to last)
        """
        # Shoudl we insert a new dimension?
        insert_dim = util._default_value(insert_dim, self.insert_dim)
        # Compute all the layers
        hs = []
        for layer in self.layers:
            h = layer(x)
            if insert_dim:
                h = operations.unsqueeze(h, d=self.dim)
            hs.append(h)
        # Concat dim
        concat_dim = self.dim
        if concat_dim < 0:
            concat_dim += len(hs[-1].dim()[0])
        # Concatenate
        return dy.concatenate(hs, d=concat_dim)
