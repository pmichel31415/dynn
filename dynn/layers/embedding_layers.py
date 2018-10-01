#!/usr/bin/env python3
"""
Embedding layers
================

For embedding discrete inputs (such as words, characters).
"""
from collections import Iterable

import numpy as np
import dynet as dy

from ..data.dictionary import Dictionary
from ..parameter_initialization import NormalInit
from ..operations import unsqueeze
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

    def __call__(self, idxs):
        """Returns the input's embedding

        If ``idxs`` is a list this returns a batch of embeddings. If it's a
        numpy array of shape ``N x b`` it returns a batch of ``b``
        ``N x embed_dim`` matrices

        Args:
            idxs (list,int): Index or list of indices to embed

        Returns:
            :py:class:`dynet.Expression`: Batch of embeddings
        """
        if not isinstance(idxs, Iterable):
            # Handle int inputs
            idxs = [idxs]
        elif not isinstance(idxs[0], Iterable):
            # List of indices
            return dy.lookup_batch(self.params, idxs, update=self.update)
        elif isinstance(idxs, np.ndarray):
            # Matrix of indices
            vecs = [dy.lookup_batch(self.params, idx, update=self.update)
                    for idx in idxs]
            return dy.concatenate([unsqueeze(vec, d=0) for vec in vecs], d=0)
        else:
            raise ValueError(
                "EmbeddingLayer only takes an int , list of ints or matrix of "
                "ints as input"
            )
