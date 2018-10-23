#!/usr/bin/env python3
"""
Transformer layers
==================
"""
import numpy as np
import dynet as dy

from ..operations import unsqueeze, seq_mask
from ..activations import relu
from ..util import conditional_dropout
from .base_layers import ParametrizedLayer
from .attention_layers import MultiHeadAttention
from .normalization_layers import LayerNorm
from .dense_layers import Affine
from .combination_layers import Sequential


def _transformer_mask(x, triu, mask, lengths, left_aligned):
    """Helper function to determine attention masks for the transformer

    Args:
        x (:py:class:`dynet.Expression`): Input (dimensions
            ``hidden_dim x L``)
        triu (bool): Upper triangular masking
        mask (:py:class:`dynet.Expression`): Mask expression
            (if ``triu`` is ``False``)
        lengths (list, optional): List of lengths (if ``mask`` is ``None``)
        left_aligned (bool, optional): Alignment (if ``mask`` is ``None``)

    Returns:
        :py:class:`dynet.Expression`, None: The actual mask
    """
    if triu:
        # Upper triangular mask
        _, L = x.dim()[0]
        # We want 0s in the upper triangualr part, so -inf in the lower
        # triangular part -> np.tril
        mask = dy.inputTensor(np.tril(np.full((L, L), -np.inf), -1))
    elif mask is not None:
        # Precomputed mask
        if len(mask.dim()[0]) == 1:
            mask = unsqueeze(mask, d=-1)
    elif lengths is not None and mask is None:
        # Recompite mask given lengths and alignment
        _, L = x.dim()[0]
        mask = seq_mask(L, lengths, 0, -np.inf, left_aligned)
    return mask


class Transformer(ParametrizedLayer):
    """Transformer layer.

    As described in `Vaswani et al. (2017) <https://arxiv.org/abs/1706.03762>`_
    This is the "encoder" side of the transformer, ie self attention only.

    Args:
        pc (:py:class:`dynet.ParameterCollection`): Parameter collection to
            hold the parameters
        hidden_dim (int): Hidden dimension (used everywhere)
        n_heads (int): Number of heads for self attention.
        activation (function, optional): MLP activation (defaults to relu).
        dropout (float, optional): Dropout rate (defaults to 0)
    """

    def __init__(
        self,
        pc,
        hidden_dim,
        n_heads,
        activation=relu,
        dropout=0.0,
    ):
        super(Transformer, self).__init__(pc, "transformer")
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
        self.layer_norm_mlp = LayerNorm(self.pc, (d, 1))

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
        triu=False,
        mask=None,
        return_att=False
    ):
        """Run the transformer layer.

        The input is expected to have dimensions ``d x L`` where ``L`` is the
        length dimension.

        Args:
            x (:py:class:`dynet.Expression`): Input (dimensions
                ``hidden_dim x L``)
            lengths (list, optional): Defaults to None. List of lengths for
                masking (used for attention)
            left_aligned (bool, optional): Defaults to True. Used for masking
            triu (bool, optional): Upper triangular self attention. Mask such
                that each position can only attend to the previous positions.
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
        mask = _transformer_mask(x, triu, mask, lengths, left_aligned)
        # Self attend
        h_att, weights = self.self_att(x, x, x, mask)
        # Dropout + residual + normalization
        x_dropped = conditional_dropout(x, self.dropout, not self.test)
        h_att = self.layer_norm_att(h_att + x_dropped, d=1)
        # MLP
        h_mlp = self.mlp(h_att + x)
        # Residual + normalization
        h_att_dropped = conditional_dropout(h_att, self.dropout, not self.test)
        h_mlp = self.layer_norm_mlp(h_mlp + h_att_dropped, d=1)
        # Return
        if return_att:
            return h_mlp, weights
        else:
            return h_mlp


class StackedTransformers(Sequential):
    """Multilayer transformer.

    Args:
        pc (:py:class:`dynet.ParameterCollection`): Parameter collection to
            hold the parameters
        n_layers (int): Number of layers
        hidden_dim (int): Hidden dimension (used everywhere)
        n_heads (int): Number of heads for self attention.
        activation (function, optional): MLP activation (defaults to relu).
        dropout (float, optional): Dropout rate (defaults to 0)
    """

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
            Transformer(pc, hidden_dim, n_heads, activation, dropout)
            for _ in range(n_layers)
        ]
        # Initialize
        super(StackedTransformers, self).__init__(*tf_layers)

    def __call__(
        self,
        x,
        lengths=None,
        left_aligned=True,
        triu=False,
        mask=None,
        return_att=False,
        return_last_only=True,
    ):
        """Run the multilayer transformer.

        The input is expected to have dimensions ``d x L`` where ``L`` is the
        length dimension.

        Args:
            x (:py:class:`dynet.Expression`): Input (dimensions
                ``hidden_dim x L``)
            lengths (list, optional): Defaults to None. List of lengths for
                masking (used for attention)
            left_aligned (bool, optional): Defaults to True. USed for masking
            triu (bool, optional): Upper triangular self attention. Mask such
                that each position can only attend to the previous positions.
            mask (:py:class:`dynet.Expression`, optional): Defaults to None.
                As an alternative to ``length``, you can pass a mask
                expression directly (useful to reuse masks accross layers)
            return_att (bool, optional): Defaults to False. Return the self
                attention weights
            return_last_only (bool, optional): Return only the output of the
                last layer (as opposed to the output of all layers).

        Returns:
            tuple, :py:class:`dynet.Expression`: The output expression (+ the
                attention weights if ``return_att`` is ``True``)
        """
        # Input has shape (d, L), B
        if len(x.dim()[0]) == 1:
            x = unsqueeze(x, d=-1)
        # Masking
        mask = _transformer_mask(x, triu, mask, lengths, left_aligned)
        # Keep track each layer's output
        outputs = ([x], [None])
        # Run all layers
        for layer in self.layers:
            new_h, weights = layer(outputs[0][-1], mask=mask, return_att=True)
            outputs[0].append(new_h)
            outputs[1].append(weights)
        # Select last output if needed
        if return_last_only:
            outputs = tuple(output[-1] for output in outputs)
        else:
            outputs = tuple(output[1:] for output in outputs)
        # Discard  attention weights if they're not needed
        if not return_att:
            outputs = outputs[0]
        return outputs


class CondTransformer(ParametrizedLayer):
    """Conditional transformer layer.

    As described in `Vaswani et al. (2017) <https://arxiv.org/abs/1706.03762>`_
    This is the "decoder" side of the transformer, ie self attention +
    attention to context.

    Args:
        pc (:py:class:`dynet.ParameterCollection`): Parameter collection to
            hold the parameters
        hidden_dim (int): Hidden dimension (used everywhere)
        cond_dim (int): Conditional dimension (dimension of the "encoder" side,
            used for attention)
        n_heads (int): Number of heads for attention.
        activation (function, optional): MLP activation (defaults to relu).
        dropout (float, optional): Dropout rate (defaults to 0)
    """

    def __init__(
        self,
        pc,
        hidden_dim,
        cond_dim,
        n_heads,
        activation=relu,
        dropout=0.0,
    ):
        super(CondTransformer, self).__init__(pc, "cond-transformer")
        # Hyper-parameters
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim
        self.activation = activation
        self.dropout = dropout
        # Sub layers
        d = hidden_dim
        dc = cond_dim
        # Self-attention
        self.self_att = MultiHeadAttention(self.pc, n_heads, d, d, d, d, d)
        self.layer_norm_self_att = LayerNorm(self.pc, (d, 1))
        # Conditional attention
        self.cond_att = MultiHeadAttention(self.pc, n_heads, d, dc, dc, d, d)
        self.layer_norm_cond_att = LayerNorm(self.pc, (d, 1))
        # MLP
        self.mlp = Sequential(
            Affine(self.pc, d, d, activation=activation),
            Affine(self.pc, d, d),
        )
        self.layer_norm_mlp = LayerNorm(self.pc, (d, 1))

    def init(self, test=True, update=False):
        self.self_att.init(test=test, update=update)
        self.layer_norm_self_att.init(test=test, update=update)
        self.cond_att.init(test=test, update=update)
        self.layer_norm_cond_att.init(test=test, update=update)
        self.mlp.init(test=test, update=update)
        self.layer_norm_mlp.init(test=test, update=update)
        self.test = test

    def __call__(
        self,
        x,
        c,
        lengths=None,
        left_aligned=True,
        mask=None,
        triu=False,
        lengths_c=None,
        left_aligned_c=True,
        mask_c=None,
        return_att=False,
    ):
        """Run the transformer layer.

        The input is expected to have dimensions ``d x L`` where ``L`` is the
        length dimension.

        Args:
            x (:py:class:`dynet.Expression`): Input (dimensions
                ``hidden_dim x L``)
            c (:py:class:`dynet.Expression`): Context (dimensions
                ``cond_dim x L``)
            lengths (list, optional): Defaults to None. List of lengths for
                masking (used for self attention)
            left_aligned (bool, optional): Defaults to True. USed for masking
                in self attention.
            mask (:py:class:`dynet.Expression`, optional): Defaults to None.
                As an alternative to ``length``, you can pass a mask
                expression directly (useful to reuse masks accross layers).
            triu (bool, optional): Upper triangular self attention. Mask such
                that each position can only attend to the previous positions.
            lengths_c (list, optional): Defaults to None. List of lengths for
                masking (used for conditional attention)
            left_aligned_c (bool, optional): Defaults to True. Used for masking
                in conditional attention.
            mask_c (:py:class:`dynet.Expression`, optional): Defaults to None.
                As an alternative to ``length_c``, you can pass a mask
                expression directly (useful to reuse masks accross layers).
            return_att (bool, optional): Defaults to False. Return the self and
                conditional attention weights

        Returns:
            tuple, :py:class:`dynet.Expression`: The output expression (+ the
                attention weights if ``return_att`` is ``True``)
        """
        # Input has shape (d, L), B
        if len(x.dim()[0]) == 1:
            x = unsqueeze(x, d=-1)
        # Context has shape (dc, l), B
        if len(c.dim()[0]) == 1:
            c = unsqueeze(c, d=-1)
        # Masking
        mask = _transformer_mask(x, triu, mask, lengths, left_aligned)
        # Masking (conditional attention)
        mask_c = _transformer_mask(c, False, mask_c, lengths_c, left_aligned_c)
        # Self attend
        h_att, self_weights = self.self_att(x, x, x, mask)
        # Dropout + residual + normalization
        x_drop = conditional_dropout(x, self.dropout, not self.test)
        h_att = self.layer_norm_self_att(h_att + x_drop, d=1)
        # Conditional attention
        h_cond, cond_weights = self.cond_att(h_att, c, c, mask_c)
        # Dropout + residual + normalization
        h_att_drop = conditional_dropout(h_att, self.dropout, not self.test)
        h_cond = self.layer_norm_cond_att(h_cond + h_att_drop, d=1)
        # MLP
        h_mlp = self.mlp(h_att + x)
        # Residual + normalization
        h_cond_drop = conditional_dropout(h_cond, self.dropout, not self.test)
        h_mlp = self.layer_norm_mlp(h_mlp + h_cond_drop, d=1)
        # Return
        if return_att:
            return h_mlp, self_weights, cond_weights
        else:
            return h_mlp


class StackedCondTransformers(Sequential):
    """Multilayer transformer.

    Args:
        pc (:py:class:`dynet.ParameterCollection`): Parameter collection to
            hold the parameters
        n_layers (int): Number of layers
        hidden_dim (int): Hidden dimension (used everywhere)
        cond_dim (int): Conditional dimension (dimension of the "encoder" side,
            used for attention)
        n_heads (int): Number of heads for self attention.
        activation (function, optional): MLP activation (defaults to relu).
        dropout (float, optional): Dropout rate (defaults to 0)
    """

    def __init__(
        self,
        pc,
        n_layers,
        hidden_dim,
        cond_dim,
        n_heads,
        activation=relu,
        dropout=0.0
    ):
        # Instatiate layers
        tf_layers = [
            CondTransformer(pc, hidden_dim, cond_dim,
                            n_heads, activation, dropout)
            for _ in range(n_layers)
        ]
        # Initialize
        super(StackedCondTransformers, self).__init__(*tf_layers)

    def __call__(
        self,
        x,
        c,
        lengths=None,
        left_aligned=True,
        mask=None,
        triu=False,
        lengths_c=None,
        left_aligned_c=True,
        mask_c=None,
        return_att=False,
        return_last_only=True,
    ):
        """Run the multilayer transformer.

        The input is expected to have dimensions ``d x L`` where ``L`` is the
        length dimension.

        Args:
            x (:py:class:`dynet.Expression`): Input (dimensions
                ``hidden_dim x L``)
            c (list): list of contexts (one per layer, each of dim
                ``cond_dim x L``). If this is not a list (but an expression),
                the same context will be used for each layer.
            lengths (list, optional): Defaults to None. List of lengths for
                masking (used for self attention)
            left_aligned (bool, optional): Defaults to True. USed for masking
                in self attention.
            mask (:py:class:`dynet.Expression`, optional): Defaults to None.
                As an alternative to ``length``, you can pass a mask
                expression directly (useful to reuse masks accross layers).
            triu (bool, optional): Upper triangular self attention. Mask such
                that each position can only attend to the previous positions.
            lengths_c (list, optional): Defaults to None. List of lengths for
                masking (used for conditional attention)
            left_aligned_c (bool, optional): Defaults to True. Used for masking
                in conditional attention.
            mask_c (:py:class:`dynet.Expression`, optional): Defaults to None.
                As an alternative to ``length_c``, you can pass a mask
                expression directly (useful to reuse masks accross layers).
            return_last_only (bool, optional): Return only the output of the
                last layer (as opposed to the output of all layers).

        Returns:
            tuple, :py:class:`dynet.Expression`: The output expression (+ the
                attention weights if ``return_att`` is ``True``)
        """
        # Input has shape (d, L), B
        if len(x.dim()[0]) == 1:
            x = unsqueeze(x, d=-1)
        # Context is a list
        if not isinstance(c, list):
            c = [c for _ in range(len(self.layers))]
        elif len(c) == 1:
            c = [c[0] for _ in range(len(self.layers))]
        # Check context size
        if len(c) != len(self.layers):
            raise ValueError(
                f"Must have either 1 or {len(self.layers)} in "
                f"{len(self.layers)}-layered conditional transformer "
                f"(got {len(c)})."
            )
        # Masking
        mask = _transformer_mask(x, triu, mask, lengths, left_aligned)
        mask_c = _transformer_mask(
            c[0],
            False,
            mask_c,
            lengths_c,
            left_aligned_c
        )
        # Keep track each layer's output
        outputs = ([x], [None], [None])
        # Run all layers
        for layer, c_ in zip(self.layers, c):
            new_h, self_weights, cond_weights = layer(
                outputs[0][-1], c_, mask=mask, mask_c=mask_c, return_att=True
            )
            outputs[0].append(new_h)
            outputs[1].append(self_weights)
            outputs[2].append(cond_weights)
        # Select last output if needed
        if return_last_only:
            outputs = tuple(output[-1] for output in outputs)
        else:
            outputs = tuple(output[1:] for output in outputs)
        # Discard  attention weights if they're not needed
        if not return_att:
            outputs = outputs[0]
        return outputs

