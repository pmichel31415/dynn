#!/usr/bin/env python3
"""
Embedding layers
================

For embedding discrete inputs (such as words, characters).
"""
import numpy as np
import dynet as dy

from ..data.dictionary import Dictionary
from ..parameter_initialization import NormalInit
from .base_layers import ParametrizedLayer


class EmbeddingLayer(ParametrizedLayer):
    """Layer for embedding elements of a dictionary

    Example:

    .. code-block:: python

        # Dictionary
        dic = dynn.data.dictionary.Dictionary(symbols=["a", "b"])
        # Parameter collection
        pc = dy.ParameterCollection()
        # Embedding layer of dimension 10
        embed = EmbeddingLayer(pc,dic, 10)
        # Initialize
        dy.renew_cg()
        embed.init()
        # Return a batch of 2 10-dimensional vectors
        vectors = embed([dic.index("b"), dic.index("a")])

    Args:
        pc (:py:class:`dynet.ParameterCollection`): Parameter collection to
            hold the parameters
        dictionary (:py:class:`dynn.data.dictionary.Dictionary`): Mapping
            from symbols to indices
        embed_dim (int): Embedding dimension
        initialization (:py:class:`dynet.PyInitializer`, optional): How
            to initialize the parameters. By default this will initialize
            to :math:`\mathcal N(0, \\frac{`}{\sqrt{\\textt{embed\_dim}}})`
    """

    def __init__(self, pc, dictionary, embed_dim, initialization=None):
        super(EmbeddingLayer, self).__init__(pc, "embedding")
        # Check input
        if not isinstance(dictionary, Dictionary):
            raise ValueError(
                "dictionary must be a dynn.data.Dictionary object"
            )
        # Dictionary and hyper-parameters
        self.dictionary = dictionary
        self.size = len(self.dictionary)
        self.embed_dim = embed_dim
        # Default init
        initialization = initialization or NormalInit(
            mean=0, std=1.0 / np.sqrt(self.size)
        )
        # Parameter shape for dynet
        if isinstance(embed_dim, (list, tuple, np.ndarray)):
            param_dim = tuple([self.size] + [dim for dim in embed_dim])
        else:
            param_dim = (self.size, embed_dim)
        # Create lookup parameter
        self.params = self.pc.add_lookup_parameters(
            param_dim,
            init=initialization,
            name="params"
        )
        # Default update parameter
        self.update = True

    def init(self, test=False, update=True):
        """Initialize the layer before performing computation

        Args:
            test (bool, optional): If test mode is set to ``True``,
                dropout is not applied (default: ``True``)
            update (bool, optional): Whether to update the parameters
                (default: ``True``)
        """
        self.test = test
        self.update = update

    def __call__(self, idx):
        """Returns the input's embedding

        If ``idx`` is a list this returns a batch of embeddings

        Args:
            idx (list,int): Index or list of indices to embed

        Returns:
            :py:class:`dynet.Expression`: Batch of embeddings
        """
        # Handle int inputs
        if isinstance(idx, int):
            idx = [idx]
        # Lookup batch
        return dy.lookup_batch(self.params, idx, update=self.update)
