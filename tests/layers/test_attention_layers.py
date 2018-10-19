#!/usr/bin/env python3

import unittest
from unittest import TestCase

import numpy as np
import dynet as dy

from dynn.layers import attention_layers


class TestMLPAttention(TestCase):

    def setUp(self):
        self.pc = dy.ParameterCollection()
        self.L = 6
        self.dk = 3
        self.dq = 4
        self.dv = 5
        self.dh = 2
        self.bsz = self.L
        self.lengths = list(range(1, self.bsz+1))
        self.mask = [[-np.inf if i >= length else 0 for length in self.lengths]
                     for i in range(self.L)]
        self.dropout = 0.1

    def test_attend(self):
        # Create compact lstm
        attend = attention_layers.MLPAttention(
            self.pc,
            self.dq,
            self.dk,
            self.dh,
            dropout=self.dropout
        )
        # Initialize computation graph
        dy.renew_cg()
        # Create inputs
        query = dy.random_uniform(self.dq, -1, 1, batch_size=self.bsz)
        keys = dy.random_uniform((self.dk, self.L), -1, 1, batch_size=self.bsz)
        values = dy.random_uniform(
            (self.dv, self.L), -1, 1, batch_size=self.bsz)
        mask = dy.inputTensor(self.mask, batched=True)
        # Initialize layer
        attend.init(test=False, update=True)
        # Run lstm cell
        y, scores = attend(query, keys, values, mask=mask)
        # Try forward/backward
        z = dy.mean_batches(dy.sum_elems(y))
        z.forward()
        z.backward(full=True)
        # Check dimension
        self.assertTupleEqual(y.dim()[0], (self.dv,))
        self.assertEqual(y.dim()[1], self.bsz)
        # Check score values
        score_sums = dy.sum_elems(scores).npvalue()
        self.assertTrue(np.allclose(score_sums, np.ones(self.bsz)))
        # Check masking
        gradients = values.gradient()
        for b, length in enumerate(self.lengths):
            grad_elem = gradients[:, :, b].T
            for pos, g_val in enumerate(grad_elem):
                is_masked = pos >= length
                zero_grad = np.allclose(g_val, 0)
                self.assertEqual(is_masked, zero_grad)


class TestBilinearAttention(TestCase):

    def setUp(self):
        self.pc = dy.ParameterCollection()
        self.L = 6
        self.l_ = 7
        self.dk = 3
        self.dq = 4
        self.dv = 5
        self.dh = 2
        self.bsz = self.L
        self.lengths = list(range(1, self.bsz+1))
        self.mask = [[-np.inf if i >= length else 0 for length in self.lengths]
                     for i in range(self.L)]
        self.dropout = 0.1

    def test_attend_dot(self):
        # Create compact lstm
        attend = attention_layers.BilinearAttention(
            self.pc,
            self.dq,
            self.dq,
            dot_product=True,
            dropout=self.dropout
        )
        # Initialize computation graph
        dy.renew_cg()
        # Create inputs
        query = dy.random_uniform(
            (self.dq, self.l_), - 1, 1, batch_size=self.bsz)
        keys = dy.random_uniform((self.dq, self.L), -1, 1, batch_size=self.bsz)
        values = dy.random_uniform(
            (self.dv, self.L), -1, 1, batch_size=self.bsz)
        mask = dy.inputTensor(self.mask, batched=True)
        # Initialize layer
        attend.init(test=False, update=True)
        # Run lstm cell
        y, scores = attend(query, keys, values, mask=mask)
        # Try forward/backward
        z = dy.mean_batches(dy.sum_elems(y))
        z.forward()
        z.backward(full=True)
        # Check dimension
        self.assertTupleEqual(y.dim()[0], (self.dv, self.l_))
        self.assertEqual(y.dim()[1], self.bsz)
        # Check score values
        score_sums = dy.sum_dim(scores, d=[0]).npvalue()
        self.assertTrue(np.allclose(score_sums, np.ones((self.l_, self.bsz))))
        # Check masking
        gradients = values.gradient()
        for b, length in enumerate(self.lengths):
            grad_elem = gradients[:, :, b].T
            for pos, g_val in enumerate(grad_elem):
                is_masked = pos >= length
                zero_grad = np.allclose(g_val, 0)
                self.assertEqual(is_masked, zero_grad)

    def test_attend_bilinear(self):
        # Create compact lstm
        attend = attention_layers.BilinearAttention(
            self.pc,
            self.dq,
            self.dk,
            dot_product=False,
            dropout=self.dropout
        )
        # Initialize computation graph
        dy.renew_cg()
        # Create inputs
        query = dy.random_uniform(
            (self.dq, self.l_), - 1, 1, batch_size=self.bsz)
        keys = dy.random_uniform((self.dk, self.L), -1, 1, batch_size=self.bsz)
        values = dy.random_uniform(
            (self.dv, self.L), -1, 1, batch_size=self.bsz)
        mask = dy.inputTensor(self.mask, batched=True)
        # Initialize layer
        attend.init(test=False, update=True)
        # Run lstm cell
        y, scores = attend(query, keys, values, mask=mask)
        # Try forward/backward
        z = dy.mean_batches(dy.sum_elems(y))
        z.forward()
        z.backward(full=True)
        # Check dimension
        self.assertTupleEqual(y.dim()[0], (self.dv, self.l_))
        self.assertEqual(y.dim()[1], self.bsz)
        # Check score values
        score_sums = dy.sum_dim(scores, d=[0]).npvalue()
        self.assertTrue(np.allclose(score_sums, np.ones((self.l_, self.bsz))))
        # Check masking
        gradients = values.gradient()
        for b, length in enumerate(self.lengths):
            grad_elem = gradients[:, :, b].T
            for pos, g_val in enumerate(grad_elem):
                is_masked = pos >= length
                zero_grad = np.allclose(g_val, 0)
                self.assertEqual(is_masked, zero_grad)


class TestMultiHeadAttention(TestCase):

    def setUp(self):
        self.pc = dy.ParameterCollection()
        self.dk = 3
        self.dq = 4
        self.dv = 5
        # Attention layer values
        self.dh = 8
        self.nh = 4
        self.do = 2
        # Lengths
        self.L = 6
        self.l_ = 7
        self.bsz = self.L
        self.lengths = list(range(1, self.bsz+1))
        self.mask = [[-np.inf if i >= length else 0 for length in self.lengths]
                     for i in range(self.L)]
        self.dropout = 0.1

    def test_attend(self):
        # Create compact lstm
        attend = attention_layers.MultiHeadAttention(
            self.pc,
            self.nh,
            self.dq,
            self.dk,
            self.dv,
            self.dh,
            self.do,
            dropout=self.dropout
        )
        # Initialize computation graph
        dy.renew_cg()
        # Create inputs
        queries = dy.random_uniform(
            (self.dq, self.l_), - 1, 1, batch_size=self.bsz)
        keys = dy.random_uniform((self.dk, self.L), -1, 1, batch_size=self.bsz)
        values = dy.random_uniform(
            (self.dv, self.L), -1, 1, batch_size=self.bsz)
        mask = dy.inputTensor(self.mask, batched=True)
        # Initialize layer
        attend.init(test=False, update=True)
        # Run lstm cell
        y, weights = attend(queries, keys, values, mask=mask)
        # Try forward/backward
        z = dy.mean_batches(dy.sum_elems(y))
        z.forward()
        z.backward(full=True)
        # Check dimension
        self.assertTupleEqual(y.dim()[0], (self.do, self.l_))
        self.assertEqual(y.dim()[1], self.bsz)
        # Check weights values
        for head in range(self.nh):
            weights_sums = dy.sum_dim(weights[head], d=[0]).npvalue()
            self.assertTrue(np.allclose(
                weights_sums, np.ones((self.l_, self.bsz))))
        # Check masking
        gradients = values.gradient()
        for b, length in enumerate(self.lengths):
            grad_elem = gradients[:, :, b].T
            for pos, g_val in enumerate(grad_elem):
                is_masked = pos >= length
                zero_grad = np.allclose(g_val, 0)
                self.assertEqual(is_masked, zero_grad)


if __name__ == '__main__':
    unittest.main()
