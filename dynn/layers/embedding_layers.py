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
        init (:py:class:`dynet.PyInitializer`, optional): How
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
        init=None,
        pad_mask=None,
        E=None,
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
        init = init or default_init
        # Parameter shape for dynet
        if isinstance(embed_dim, (list, tuple, np.ndarray)):
            dim = tuple([self.size] + [d for d in embed_dim])
        else:
            dim = (self.size, embed_dim)
        # Create lookup parameter
        if E is not None and isinstance(E, dy.Parameters):
            self.is_lookup = False
            self.add_parameters("E", dim, param=E)
        else:
            self.is_lookup = True
            self.add_lookup_parameters("E", dim, lookup_param=E, init=init)

    def _lookup(self, idx):
        if self.is_lookup:
            return dy.lookup_batch(self.E, idx, update=self.update)
        else:
            return dy.pick_batch(self.E, idx)

    @property
    def weights(self):
        """Numpy array containing the embeddings

        The first dimension is the lookup dimension """
        return self.E.as_array()

    def __call__(self, idxs, length_dim=0):
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
            d = self.embed_dim
            L, bsz = idxs.shape
            flat_embeds = self._lookup(idxs.flatten(order="F"))
            embeds = dy.reshape(flat_embeds, (d, L), batch_size=bsz)
            if length_dim == 0:
                embeds = dy.transpose(embeds)
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
                mask = unsqueeze(mask, d=1-length_dim)
            # Apply the mask
            embeds = dy.cmult(1-mask, embeds) + self.pad_mask * mask

        return embeds
