#!/usr/bin/env python3
"""
Transformer layers
==================
"""
import numpy as np

from ..operations import unsqueeze, seq_mask
from ..activations import relu
from ..util import conditional_dropout
from .base_layers import ParametrizedLayer
from .attention_layers import MultiHeadAttention
from .normalization_layers import LayerNorm
from .dense_layers import Affine
from .combination_layers import Sequential


class TransformerLayer(ParametrizedLayer):
    """Transformer layer.

    As described in `Vaswani et al. (2017) <https://arxiv.org/abs/1706.03762>`_
    This is the "encoder" side of the transformer, ie self attention only.

    Args:
        pc (:py:class:`dynet.ParameterCollection`): Parameter collection to
            hold the parameters
        query_dim (int): Queries dimension
        key_dim (int): Keys dimension
        hidden_dim (int): Hidden dimension of the MLP
        activation (function, optional): MLP activation (defaults to tanh).
        dropout (float, optional): Attention dropout (defaults to 0)

    """

    def __init__(
        self,
        pc,
        hidden_dim,
        n_heads,
        activation=relu,
        dropout=0.0,
    ):
        super(TransformerLayer, self).__init__(pc, "mlp-attention")
        # Hyper-parameters
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.dropout = dropout
        # Sub layers
        d = hidden_dim
        # Self-attention
        self.self_att = MultiHeadAttention(self.pc, n_heads, d, d, d, d, d)
        self.layer_norm_att = LayerNorm(self.pc, (d, 1))
        # MLP
        self.mlp = Sequential(
            Affine(self.pc, d, d, activation=activation),
            Affine(self.pc, d, d),
        )
        self.layer_norm_mlp = LayerNorm(self.pc, (1, d))

    def init(self, test=True, update=False):
        self.self_att.init(test=test, update=update)
        self.layer_norm_att.init(test=test, update=update)
        self.mlp.init(test=test, update=update)
        self.layer_norm_mlp.init(test=test, update=update)
        self.test = test

    def __call__(
        self,
        x,
        lengths=None,
        left_aligned=True,
        mask=None,
        return_att=False
    ):
        """Run the transformer layer.

        The input is expected to have dimensions ``d x L`` where ``L`` is the
        length dimension.

        Args:
            x (::): [description]
            lengths (list, optional): Defaults to None. List of lengths for
                masking (used for attention)
            left_aligned (bool, optional): Defaults to True. USed for masking
            mask (:py:class:`dynet.Expression`, optional): Defaults to None.
                As an alternative to ``length``, you can pass a mask
                expression directly (useful to reuse masks accross layers)
            return_att (bool, optional): Defaults to False. Return the self
                attention weights

        Returns:
            tuple, :py:class:`dynet.Expression`: The output expression (+ the
                attention weights if ``return_att`` is ``True``)
        """
        # Input has shape (d, L), B
        if len(x.dim()[0]) == 1:
            x = unsqueeze(x, d=-1)
        # Masking
        if mask is not None:
            if len(mask.dim()[0]) == 1:
                mask = unsqueeze(mask, d=-1)
        elif lengths is not None and mask is None:
            _, L = x.dim()[0]
            mask = seq_mask(L, lengths, 0, -np.inf, left_aligned)
        # Self attend
        h_att, weights = self.self_att(x, x, x, mask)
        # Dropout + residual + normalization
        x_dropped = conditional_dropout(x, self.dropout, not self.test)
        h_att = self.layer_norm_att(h_att + x_dropped, d=1)
        # MLP
        h_mlp = self.mlp(h_att + x)
        # Residual + normalization
        h_att_dropped = conditional_dropout(h_att, self.dropout, not self.test)
        h_mlp = self.layer_norm_att(h_mlp + h_att_dropped, d=1)
        # Return
        if return_att:
            return h_mlp, weights
        else:
            return h_mlp


class StackedTransformerLayers(Sequential):

    def __init__(
        self,
        pc,
        n_layers,
        hidden_dim,
        n_heads,
        activation=relu,
        dropout=0.0
    ):
        # Instatiate layers
        tf_layers = [
            TransformerLayer(pc, hidden_dim, n_heads, activation, dropout)
            for _ in range(n_layers)
        ]
        # Initialize
        super(StackedTransformerLayers, self).__init__(*tf_layers)

    def __call__(
        self,
        x,
        lengths=None,
        left_aligned=True,
        mask=None,
        return_att=False,
        return_last_only=True,
    ):
        # Input has shape (d, L), B
        if len(x.dim()[0]) == 1:
            x = unsqueeze(x, d=-1)
        # Masking
        if mask is not None:
            if len(mask.dim()[0]) == 1:
                mask = unsqueeze(mask, d=-1)
        elif lengths is not None and mask is None:
            _, L = x.dim()[0]
            mask = seq_mask(L, lengths, 0, -np.inf, left_aligned)
        # Keep track each layer's output
        outputs = ([x], [None])
        # Run all layers
        for layer in self.layers:
            new_h, weights = layer(outputs[0][-1], mask=mask, return_att=True)
            outputs[0].append(new_h)
            outputs[1].append(weights)
        # Discard weights if they're not needed
        if not return_att:
            outputs = outputs[0]
        # Return
        if return_last_only:
            return outputs[-1]
        else:
            return outputs[1:]
