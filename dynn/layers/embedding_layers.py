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


class Embeddings(ParametrizedLayer):
    """Layer for embedding elements of a dictionary

    Example:

    .. code-block:: python

        # Dictionary
        dic = dynn.data.dictionary.Dictionary(symbols=["a", "b"])
        # Parameter collection
        pc = dy.ParameterCollection()
        # Embedding layer of dimension 10
        embed = Embeddings(pc,dic, 10)
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
        pad_mask (float, optional): If provided, embeddings of the
            ``dictionary.pad_idx`` index will be masked with this value
    """

    def __init__(
        self,
        pc,
        dictionary,
        embed_dim,
        initialization=None,
        pad_mask=None,
        params=None,
    ):
        super(Embeddings, self).__init__(pc, "embedding")
        # Check input
        if not isinstance(dictionary, Dictionary):
            raise ValueError(
                "dictionary must be a dynn.data.Dictionary object"
            )
        # Dictionary and hyper-parameters
        self.dictionary = dictionary
        self.size = len(self.dictionary)
        self.embed_dim = embed_dim
        self.pad_mask = pad_mask
        # Default init
        default_init = NormalInit(std=1/np.sqrt(self.embed_dim))
        initialization = initialization or default_init
        # Parameter shape for dynet
        if isinstance(embed_dim, (list, tuple, np.ndarray)):
            param_dim = tuple([self.size] + [dim for dim in embed_dim])
        else:
            param_dim = (self.size, embed_dim)
        # Create lookup parameter
        self.params = params or self.pc.add_lookup_parameters(
            param_dim,
            init=initialization,
            name="params"
        )
        self.is_lookup = isinstance(self.params, dy.LookupParameters)
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
        if not self.is_lookup:
            self.params_e = self.params.expr(update)
        self.test = test
        self.update = update

    def _lookup(self, idx):
        if self.is_lookup:
            return dy.lookup_batch(self.params, idx, update=self.update)
        else:
            return dy.pick_batch(self.params_e, idx)
        
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
        idxs = np.asarray(idxs, dtype=int)
        if len(idxs.shape) == 1:
            # List of indices
            embeds = self._lookup(idxs)
        elif len(idxs.shape) == 2:
            # Matrix of indices
            vecs = [self._lookup(idx) for idx in idxs]
            embeds = dy.concatenate([unsqueeze(vec, d=0) for vec in vecs], d=0)
        else:
            raise ValueError(
                "Embeddings only takes an int , list of ints or matrix of "
                "ints as input"
            )

        # Masking
        if self.pad_mask is not None:
            is_padding = (idxs == self.dictionary.pad_idx).astype(int)
            mask = dy.inputTensor(is_padding, batched=True)
            # Insert a dimension of size 1 for the embedding dimension
            # This is automatic when the input is only 1 index per batch
            # element
            if len(idxs.shape) == 2:
                mask = unsqueeze(mask, d=-1)
            # Apply the mask
            embeds = dy.cmult(1-mask, embeds) + self.pad_mask * mask

        return embeds
