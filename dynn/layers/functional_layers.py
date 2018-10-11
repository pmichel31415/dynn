#!/usr/bin/env python3
"""
Functional layers
=================
"""

import dynet as dy

from .base_layers import BaseLayer
from ..activations import identity


class ConstantLayer(BaseLayer):
    """This is the "zero"-ary layer.


    .. code-block:: python

        # Takes in numbers
        ConstantLayer(5)() == dy.inputTensor([5])
        # Or lists
        ConstantLayer([5, 6])() == dy.inputTensor([5, 6])
        # Or numpy arrays
        ConstantLayer(np.ones((10, 12)))() == dy.inputTensor(np.ones((10, 12)))

    Args:
        constant (number, np.ndarray): The constant. It must be a type
            that can be turned into a :py:class:`dynet.Expression`
    """

    def __init__(self, constant):
        super(ConstantLayer, self).__init__(f"constant_{constant}")
        if isinstance(constant, (int, float)):
            constant = [constant]
        self.constant_val = constant
        self.constant = dy.inputTensor(self.constant_val)

    def init(self, *args, **kwargs):
        self.constant = dy.inputTensor(self.constant_val)

    def __call__(self, *args, **kwargs):
        return self.constant


class LambdaLayer(BaseLayer):
    """This layer applies an arbitrary function to its input.

    .. code-block:: python

        LambdaLayer(f)(x) == f(x)

    This is useful if you want to wrap activation functions as layers. The
    unary operation should be a function taking :py:class:`dynet.Expression` to
    :py:class:`dynet.Expression`.

    You shouldn't use this to stack layers though, ``op`` oughtn't be a layer.
    If you want to stack layers, use
    :py:class:`combination_layers.Sequential`.

    Args:
        layer (:py:class:`base_layers.BaseLayer`): The layer to which output
            you want to apply the unary operation.
        binary_operation (function): A unary operation on
            :py:class:`dynet.Expression` objects
    """

    def __init__(self, function):
        super(LambdaLayer, self).__init__(
            f"lambda_{function.__name__}"
        )
        self.function = function

    def __call__(self, *args, **kwargs):
        """Returns ``function(*args, **kwargs)``"""
        return self.function(*args, **kwargs)


class IdentityLayer(LambdaLayer):
    """The identity layer does literally nothing

    .. code-block:: python

        IdentityLayer()(x) == x

    It passes its input directly as the output. Still, it can be useful to
    express more complicated layers like residual connections.
    """

    def __init__(self):
        super(IdentityLayer, self).__init__(identity)


class UnaryOpLayer(BaseLayer):
    """This layer wraps a unary operation on another layer.

    .. code-block:: python

        UnaryOpLayer(layer, op)(x) == op(layer(x))

    This is a shorter way of writing:

    .. code-block:: python

        UnaryOpLayer(layer, op)(x) == Sequential(layer, LambdaLayer(op))

    You shouldn't use this to stack layers though, ``op`` oughtn't be a layer.
    If you want to stack layers, use
    :py:class:`combination_layers.Sequential`.

    Args:
        layer (:py:class:`base_layers.BaseLayer`): The layer to which output
            you want to apply the unary operation.
        binary_operation (function): A unary operation on
            :py:class:`dynet.Expression` objects
    """

    def __init__(self, layer, unary_operation):
        super(UnaryOpLayer, self).__init__(
            f"unary_{unary_operation.__name__}"
        )
        if isinstance(unary_operation, BaseLayer):
            raise ValueError("The unary operation cannot be a Layer")
        self.layer = layer
        self.unary_operation = unary_operation

    def init(self, *args, **kwargs):
        """Initialize the wrapped layer"""
        self.layer.init(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        """Returns ``unary_operation(layer(*args, **kwargs))``"""
        return self.unary_operation(self.layer(*args, **kwargs))


class NegationLayer(UnaryOpLayer):
    """Negates the output of another layer:

    .. code-block:: python

        NegationLayer(layer)(x) == - layer(x)

    It can also be used with the `-` syntax directly:

    .. code-block:: python

        negated_layer = - layer
        # is the same as
        negated_layer = NegationLayer(layer)

    Args:
        layer (:py:class:`base_layers.BaseLayer`): The layer to which output
            you want to apply the negation.
    """

    def __init__(self, layer):
        super(NegationLayer, self).__init__(layer, dy.Expression.__neg__)


# Add to BaseLayer
def negate_layer(layer):
    return NegationLayer(layer)


BaseLayer.__neg__ = negate_layer


class BinaryOpLayer(BaseLayer):
    """This layer wraps two layers with a binary operation.

    .. code-block:: python

        BinaryOpLayer(layer1, layer2, op)(x) == op(layer1(x), layer2(x))

    This is useful to express the addition of two layers as another layer.

    Args:
        layer1 (:py:class:`base_layers.BaseLayer`): First layer
        layer2 (:py:class:`base_layers.BaseLayer`): Second layer
        binary_operation (function): A binary operation on
            :py:class:`dynet.Expression` objects
    """

    def __init__(self, layer1, layer2, binary_operation):
        super(BinaryOpLayer, self).__init__(
            f"binary_{binary_operation.__name__}"
        )
        if isinstance(binary_operation, BaseLayer):
            raise ValueError("The unary operation cannot be a Layer")
        self.layer1 = layer1
        self.layer2 = layer2
        self.binary_operation = binary_operation

    def init(self, *args, **kwargs):
        self.layer1.init(*args, **kwargs)
        self.layer2.init(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        """Execute forward pass"""
        output1 = self.layer1(*args, **kwargs)
        output2 = self.layer2(*args, **kwargs)
        return self.binary_operation(output1, output2)


class AdditionLayer(BinaryOpLayer):
    """Addition of two layers.

    This is the layer returned by the addition syntax:

    .. code-block:: python

        AdditionLayer(layer1, layer2)(x) == layer1(x) + layer2(x)
        # is the same thing as
        add_1_2 = layer1 + layer2
        add_1_2(x) == layer1(x) + layer2(x)

    Args:
        layer1 (:py:class:`base_layers.BaseLayer`): First layer
        layer2 (:py:class:`base_layers.BaseLayer`): Second layer
    """

    def __init__(self, layer1, layer2):
        super(AdditionLayer, self).__init__(
            layer1, layer2, dy.Expression.__add__
        )


# Add to BaseLayer
def add_layer(layer1, layer2):
    return AdditionLayer(layer1, layer2)


BaseLayer.__add__ = add_layer
BaseLayer.__radd__ = add_layer


class SubstractionLayer(BinaryOpLayer):
    """Substraction of two layers.

    This is the layer returned by the substraction syntax:

    .. code-block:: python

        SubstractionLayer(layer1, layer2)(x) == layer1(x) - layer2(x)
        # is the same thing as
        add_1_2 = layer1 - layer2
        add_1_2(x) == layer1(x) - layer2(x)

    Args:
        layer1 (:py:class:`base_layers.BaseLayer`): First layer
        layer2 (:py:class:`base_layers.BaseLayer`): Second layer
    """

    def __init__(self, layer1, layer2):
        super(SubstractionLayer, self).__init__(
            layer1, layer2, dy.Expression.__sub__
        )


# Add to BaseLayer
def sub_layer(layer1, layer2):
    return SubstractionLayer(layer1, layer2)


BaseLayer.__sub__ = sub_layer


class CmultLayer(BinaryOpLayer):
    """Coordinate-wise multiplication of two layers.

    .. code-block:: python

        CmultLayer(layer1, layer2)(x) == dy.cmult(layer1(x), layer2(x))

    Args:
        layer1 (:py:class:`base_layers.BaseLayer`): First layer
        layer2 (:py:class:`base_layers.BaseLayer`): Second layer
    """

    def __init__(self, layer1, layer2):
        super(CmultLayer, self).__init__(
            layer1, layer2, dy.cmult
        )
