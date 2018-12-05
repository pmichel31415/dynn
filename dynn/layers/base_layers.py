#!/usr/bin/env python3
"""
Base layer
==========
"""
import dynet as dy


class BaseLayer(object):
    """Base layer interface"""

    def __init__(self, name):
        self.name = name
        self.test = True
        self.update = False

    def init(self, test=True, update=False):
        """Initialize the layer before performing computation

        For example setup dropout, freeze some parameters, etc...
        """
        self.test = test
        self.update = update
        for sublayer in self.sublayers.values():
            sublayer.init(test=test, update=update)
        self.init_layer(test=test, update=update)

    def init_layer(self, test=True, update=False):
        """Initializes only this layer's parameters (not recursive)
        This needs to be implemented for each layer """
        pass

    def __call__(self, *args, **kwargs):
        """Execute forward pass"""
        raise NotImplementedError()

    @property
    def sublayers(self):
        """Returns all attributes of the layer which are layers themselves"""
        _sublayers = {}
        for name, attr in self.__dict__.items():
            if not name.startswith("__") and isinstance(attr, BaseLayer):
                _sublayers[name] = attr
        return _sublayers


class ParametrizedLayer(BaseLayer):
    """This is the base class for layers with trainable parameters

    When implementing a `ParametrizedLayer`, use ``self.add_parameters`` /
    ``self.add_lookup_parameters`` to add parameters to the layer.
    """

    def __init__(self, pc, name):
        """Creates a subcollection for this layer with a custom name"""
        super(ParametrizedLayer, self).__init__(name)
        self.pc = pc.add_subcollection(name=name)
        self._parameters = {}
        self._lookup_parameters = {}

    def add_parameters(
        self,
        name,
        dim,
        param=None,
        init=None,
        device="",
        scale=1.0,
        mean=0.0,
        std=1.0
    ):
        """This adds a parameter to this layer's ParameterCollection.

        The layer will have 1 new attribute: ``self.[name]`` which will contain
        the expression for this parameter (which you should use in
        ``__call__``).

        You can provide an existing parameter with the `param` argument, in
        which case this parameter will be reused.

        The other arguments are the same as
        :py:class:`dynet.ParameterCollection.add_parameters`
        """
        # Check for name availability
        if name in self.parameters:
            raise ValueError(
                f"Layer {self.__class__.__name__} already has a "
                f"parameter named {name}"
            )
        if param is not None:
            # Check that the provided value is a parameter
            if not isinstance(param, dy.Parameters):
                raise ValueError(
                    f"Specified value for parameter \"{name}\" in "
                    f"{self.__class__.__name__} is not a dy.Parameters object"
                )
            # Check that it has the same shape as what we would've gotten
            if not dim == param.shape():
                raise ValueError(
                    f"Mismatch between provided parameter of shape"
                    f" {param.shape()} and expected shape {dim}."
                )
        else:
            param = self.pc.add_parameters(
                dim=dim,
                init=init,
                name=name,
                device=device,
                scale=scale,
                mean=mean,
                std=std,
            )
        self._parameters[name] = param
        setattr(self, f"{name}", dy.Expression())

    def add_lookup_parameters(
        self,
        name,
        dim,
        lookup_param=None,
        init=None,
        device="",
        scale=1.0,
        mean=0.0,
        std=1.0
    ):
        """This adds a parameter to this layer's parametercollection

        The layer will have 1 new attribute: ``self.[name]`` which will contain
        the lookup parameter object (which you should use in ``__call__``).

        You can provide an existing lookup parameter with the ``lookup_param``
        argument, in which case this parameter will be reused.

        The other arguments are the same as
        :py:class:`dynet.ParameterCollection.add_lookup_parameters`
        """
        # Check for name availability
        if name in self.lookup_parameters:
            raise ValueError(
                f"Layer {self.__class__.__name__} already has a lookup "
                f"parameter named {name}"
            )

        if lookup_param is not None:
            # Check that the provided value is a parameter
            if not isinstance(lookup_param, dy.LookupParameters):
                raise ValueError(
                    f"Specified value for parameter \"{name}\" in "
                    f"{self.__class__.__name__} is not a dy.LookupParameters "
                    f"object"
                )
            # Check that it has the same shape as what we would've gotten
            if not dim == lookup_param.shape():
                raise ValueError(
                    f"Mismatch between provided parameter of shape"
                    f" {lookup_param.shape()} and expected shape {dim}."
                )
        else:
            lookup_param = self.pc.add_lookup_parameters(
                dim=dim,
                init=init,
                name=name,
                device=device,
                scale=scale,
                mean=mean,
                std=std,
            )
        self._lookup_parameters[name] = lookup_param
        setattr(self, f"{name}", lookup_param)

    def init_layer(self, test=True, update=False):
        """Initializes only this layer's parameters (not recursive)
        This needs to be implemented for each layer """
        for name, param in self.parameters.items():
            setattr(self, name, param.expr(update))

    @property
    def parameters(self):
        """Return all parameters specific to this layer"""
        return self._parameters

    @property
    def lookup_parameters(self):
        """Return all lookup parameters specific to this layer"""
        return self._lookup_parameters
