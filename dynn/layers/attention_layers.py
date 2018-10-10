#!/usr/bin/env python3
"""
Pooling layers
==============
"""

import dynet as dy
from ..operations import unsqueeze, squeeze
from ..activations import tanh
from ..util import conditional_dropout
from .base_layers import ParametrizedLayer


class MLPAttentionLayer(ParametrizedLayer):
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
        super(MLPAttentionLayer, self).__init__(pc, "mlp-attention")
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
            raise ValueError("#keys != #values in MLPAttentionLayer")
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
