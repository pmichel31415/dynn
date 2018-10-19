#!/usr/bin/env python3
"""
Attention layers
================
"""
import numpy as np
import dynet as dy

from ..operations import unsqueeze, squeeze
from ..activations import tanh
from ..parameter_initialization import UniformInit
from ..util import conditional_dropout
from .base_layers import ParametrizedLayer


class MLPAttention(ParametrizedLayer):
    """Multilayer Perceptron based attention

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
        query_dim,
        key_dim,
        hidden_dim,
        activation=tanh,
        dropout=0.0,
    ):
        super(MLPAttention, self).__init__(pc, "mlp-attention")
        # Hyper-parameters
        self.key_dim = key_dim
        self.query_dim = query_dim
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.dropout = dropout
        # Parameters
        self.Wq_p = self.pc.add_parameters((hidden_dim, query_dim), name="Wq")
        self.Wk_p = self.pc.add_parameters((hidden_dim, key_dim), name="Wk")
        self.b_p = self.pc.add_parameters((hidden_dim,), name="b")
        self.V_p = self.pc.add_parameters((1, hidden_dim), name="V")

    def init(self, test=True, update=False):
        self.Wq = self.Wq_p.expr(update)
        self.Wk = self.Wk_p.expr(update)
        self.b = self.b_p.expr(update)
        self.V = self.V_p.expr(update)
        self.test = test

    def __call__(self, query, keys, values, mask=None):
        """Compute attention scores and return the pooled value

        This returns both the pooled value and the attention score. You can
        specify an **additive** mask when some values are not to be attended to
        (eg padding).

        Args:
            query (:py:class:`dynet.Expression`): Query vector of size
                ``(dq,), B``
            keys (:py:class:`dynet.Expression`): Key vectors of size
                ``(dk, L), B``
            values (:py:class:`dynet.Expression`): Value vectors of size
                ``(dv, L), B``
            mask (:py:class:`dynet.Expression`, optional): Additive mask
                expression

        Returns:
            tuple: ``pooled_value, scores``, of size ``(dv,), B`` and
                ``(L,), B`` respectively
        """

        # query has shape (dq,), B
        # keys has shape (dk, L), B
        if len(keys.dim()[0]) == 1:
            keys = unsqueeze(keys, d=-1)
        # values has shape (dv, L), B
        if len(values.dim()[0]) == 1:
            values = unsqueeze(values, d=-1)
        # Check that keys length == queries length
        L = keys.dim()[0][1]
        if L != values.dim()[0][1]:
            raise ValueError("#keys != #values in MLPAttention")
        # Dropout
        query = conditional_dropout(query, self.dropout, not self.test)
        keys = conditional_dropout(keys, self.dropout, not self.test)
        # Compute hidden state
        h_query = dy.affine_transform([self.b, self.Wq, query])
        h = unsqueeze(h_query, d=1) + self.Wk * keys
        # Logits
        logits = squeeze(self.V * tanh(h), d=0)
        # Masking maybe
        if mask is not None:
            logits += mask
        # Scores
        scores = dy.softmax(logits)
        # Compute average value
        pooled_value = values * scores
        return pooled_value, scores


class BilinearAttention(ParametrizedLayer):
    """Bilinear attention layer.

    Here the scores are computed according to

    .. math::

        \\alpha_{ij}=q_i^\intercal A k_j

    Where :math:`q_i,k_j` are the ith query and jth key respectively. If
    ``dot_product`` is set to ``True`` this is replaced by:

    .. math::

        \\alpha_{ij}=\\frac 1 {\sqrt{d}} q_i^\intercal k_j

    Where :math:`d` is the dimension of the keys and queries.

    Args:
        pc (:py:class:`dynet.ParameterCollection`): Parameter collection to
            hold the parameters
        query_dim (int): Queries dimension
        key_dim (int): Keys dimension
        dot_product (bool, optional): Compute attention with the dot product
            only (no weight matrix). The requires that ``query_dim==key_dim``.
        dropout (float, optional): Attention dropout (defaults to 0)
        A_p (:py:class:`dynet.Parameters`, optional): Specify the weight matrix
            directly.

    """

    def __init__(
        self,
        pc,
        query_dim,
        key_dim,
        dot_product=False,
        dropout=0.0,
        A_p=None,
    ):
        super(BilinearAttention, self).__init__(pc, "dot-attention")
        # Hyper-parameters
        self.key_dim = key_dim
        self.query_dim = query_dim
        self.dot_product = dot_product
        self.dropout = dropout
        # Parameters
        if self.dot_product:
            if key_dim != query_dim:
                raise ValueError("")
        if not self.dot_product:
            init_scale = np.sqrt(3/(self.key_dim * self.query_dim))
            self.A_p = A_p or self.pc.add_parameters(
                (self.key_dim, self.query_dim),
                name="A",
                init=UniformInit(init_scale),
            )

    def init(self, test=True, update=False):
        if not self.dot_product:
            self.A = self.A_p.expr(update)
        self.test = test

    def __call__(self, query, keys, values, mask=None):
        """Compute attention scores and return the pooled value.

        This returns both the pooled value and the attention score. You can
        specify an **additive** mask when some values are not to be attended to
        (eg padding).

        Args:
            query (:py:class:`dynet.Expression`): Query vector of size
                ``(dq, l), B``
            keys (:py:class:`dynet.Expression`): Key vectors of size
                ``(dk, L), B``
            values (:py:class:`dynet.Expression`): Value vectors of size
                ``(dv, L), B``
            mask (:py:class:`dynet.Expression`, optional): Additive mask
                expression for the source side (size ``(L,), B``)

        Returns:
            tuple: ``pooled_value, scores``, of size ``(dv,), B`` and
                ``(L,), B`` respectively
        """

        # query has shape (dq, l), B
        if len(query.dim()[0]) == 1:
            query = unsqueeze(query, d=-1)
        # keys has shape (dk, L), B
        if len(keys.dim()[0]) == 1:
            keys = unsqueeze(keys, d=-1)
        # values has shape (dv, L), B
        if len(values.dim()[0]) == 1:
            values = unsqueeze(values, d=-1)
        # Check that keys length == queries length
        L = keys.dim()[0][1]
        if L != values.dim()[0][1]:
            raise ValueError("#keys != #values in MLPAttention")
        # Dropout
        query = conditional_dropout(query, self.dropout, not self.test)
        keys = conditional_dropout(keys, self.dropout, not self.test)
        # Compute logits (they have shape L x l)
        if self.dot_product:
            logits = dy.transpose(keys) * query / np.sqrt(self.key_dim)
        else:
            logits = dy.transpose(keys) * (self.A * query)
        # Masking maybe
        if mask is not None:
            # Mask should have shape L x 1
            if len(mask.dim()[0]) == 1:
                mask = unsqueeze(mask, d=-1)
            logits += mask
        # Scores
        scores = dy.softmax(logits, d=0)
        # Compute average value (shape dv x l)
        pooled_value = values * scores
        return pooled_value, scores


class MultiHeadAttention(ParametrizedLayer):
    """Multi headed attention layer.

    This functions like dot product attention

    .. math::

        \\alpha_{ij}=\\frac 1 {\sqrt{d}} q_i^\intercal k_j

    Except the key, query and values are split into multiple ``heads``.

    Args:
        pc (:py:class:`dynet.ParameterCollection`): Parameter collection to
            hold the parameters
        n_heads (int): Number of heads
        query_dim (int): Dimension of queries
        key_dim (int): Dimension of keys
        values_dim (int): Dimension of values
        hidden_dim (int): Hidden dimension (must be a multiple of ``n_heads``)
        out_dim (bool, optional): Output dimension
        dropout (float, optional): Attention dropout (defaults to 0)
        Wq_p (:py:class:`dynet.Parameters`, optional): Specify the queries
            projection matrix directly.
        Wk_p (:py:class:`dynet.Parameters`, optional): Specify the keys
            projection matrix directly.
        Wv_p (:py:class:`dynet.Parameters`, optional): Specify the values
            projection matrix directly.
        Wo_p (:py:class:`dynet.Parameters`, optional): Specify the output
            projection matrix directly.

    """

    def __init__(
        self,
        pc,
        n_heads,
        query_dim,
        key_dim,
        value_dim,
        hidden_dim,
        out_dim,
        dropout=0.0,
        Wq_p=None,
        Wk_p=None,
        Wv_p=None,
        Wo_p=None,
    ):
        super(MultiHeadAttention, self).__init__(pc, "dot-attention")
        # Hyper-parameters
        self.n_heads = n_heads
        self.key_dim = key_dim
        self.query_dim = query_dim
        self.value_dim = value_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.dropout = dropout
        # Check that att_dim and n_heads are compatible
        if not self.hidden_dim % self.n_heads == 0:
            raise ValueError(
                f"Hidden dim ({self.hidden_dim}) must be a multiple of the "
                f"number of heads ({self.n_heads})"
            )
        self.head_dim = self.hidden_dim // self.n_heads
        # Parameters
        self.Wq_p = Wq_p or self.pc.add_parameters(
            (self.hidden_dim, self.query_dim),
            name="Wq",
        )
        self.Wk_p = Wk_p or self.pc.add_parameters(
            (self.hidden_dim, self.key_dim),
            name="Wk",
        )
        self.Wv_p = Wv_p or self.pc.add_parameters(
            (self.hidden_dim, self.value_dim),
            name="Wv",
        )
        self.Wo_p = Wo_p or self.pc.add_parameters(
            (self.out_dim, self.hidden_dim),
            name="Wo",
        )

    def init(self, test=True, update=False):
        self.Wq = self.Wq_p.expr(update)
        self.Wk = self.Wk_p.expr(update)
        self.Wv = self.Wv_p.expr(update)
        self.Wo = self.Wo_p.expr(update)
        self.test = test

    def __call__(self, queries, keys, values, mask=None):
        """Compute attention weightss and return the pooled value.

        This expects the queries, keys and values to have dimensions
        ``dq x l``, ``dk x L``, ``dv x L`` respectively.

        Returns both the pooled value and the attention weights (list of
        weights, one per head). You can specify an **additive** mask when
        some values are not to be attended to (eg padding).

        Args:
            queries (:py:class:`dynet.Expression`): Query vector of size
                ``(dq, l), B``
            keys (:py:class:`dynet.Expression`): Key vectors of size
                ``(dk, L), B``
            values (:py:class:`dynet.Expression`): Value vectors of size
                ``(dv, L), B``
            mask (:py:class:`dynet.Expression`, optional): Additive mask
                expression for the source side (size ``(L,), B``)

        Returns:
            tuple: ``pooled_value, scores``, of size ``(dv,), B`` and
                ``(L,), B`` respectively
        """

        # queries has shape (dq, l), B
        if len(queries.dim()[0]) == 1:
            queries = unsqueeze(queries, d=-1)
        # keys has shape (dk, L), B
        if len(keys.dim()[0]) == 1:
            keys = unsqueeze(keys, d=-1)
        # values has shape (dv, L), B
        if len(values.dim()[0]) == 1:
            values = unsqueeze(values, d=-1)
        # Check that keys length == values length
        L = keys.dim()[0][1]
        if L != values.dim()[0][1]:
            raise ValueError("#keys != #values in MLPAttention")
        # Dropout
        queries = conditional_dropout(queries, self.dropout, not self.test)
        keys = conditional_dropout(keys, self.dropout, not self.test)
        values = conditional_dropout(values, self.dropout, not self.test)
        # Project all the things
        hq = self.Wq * queries
        hk = self.Wk * keys
        hv = self.Wv * values
        # Compute weights for each head (they have shape L x l)
        weights = []
        for head in range(self.n_heads):
            # Slice for each head (dh//nh x l and dh//nh x L respectively)
            head_hq = hq[head * self.head_dim:(head + 1) * self.head_dim]
            head_hk = hk[head * self.head_dim:(head + 1) * self.head_dim]
            # Logits (L x l)
            logits = dy.transpose(head_hk) * head_hq
            logits /= np.sqrt(self.head_dim)
            # Masking maybe
            if mask is not None:
                # Mask should have shape L x l
                if len(mask.dim()[0]) == 1:
                    mask = unsqueeze(mask, d=-1)
                logits += mask
            # Weights
            weights.append(dy.softmax(logits, d=0))
        # Compute values
        head_values = []
        for head in range(self.n_heads):
            # Slice each head's value
            head_hv = hv[head * self.head_dim:(head + 1) * self.head_dim]
            # Average with the head weights
            head_values.append(head_hv * weights[head])
        # Concatenate and apply the last linear transform
        pooled_value = self.Wo * dy.concatenate(head_values, d=0)
        # Return pooled value and weights
        return pooled_value, weights
