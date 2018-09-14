#!/usr/bin/env python3

from unittest import TestCase

import numpy as np
import dynet as dy

from dynn import util


class TestUtil(TestCase):

    def setUp(self):
        self.N = 20
        self.bsz = 4
        self.d = 10
        self.dropout_rate = 0.5

    def test_list_to_matrix(self):
        dy.renew_cg()
        # Input a list of vectors
        matrix = util.list_to_matrix([
            dy.random_uniform(self.d, 0, 1, batch_size=self.bsz)
            for _ in range(self.N)
        ])
        # Check dimension
        self.assertTupleEqual(matrix.dim()[0], (self.d, self.N))
        self.assertEqual(matrix.dim()[1], self.bsz)
        # Feeding a matrix shouldn't be a problem
        util.list_to_matrix(matrix)

    def test_matrix_to_image(self):
        dy.renew_cg()
        # Input matrix
        M = dy.inputTensor(np.random.rand(
            self.d, self.N, self.bsz), batched=True
        )
        # Call on vector
        image = util.matrix_to_image(M[0])
        # Check dimensions
        self.assertSequenceEqual(image.dim()[0], (self.N, 1, 1))
        self.assertEqual(image.dim()[1], self.bsz)
        # Call on matrix
        image = util.matrix_to_image(M)
        # Check dimensions
        self.assertTupleEqual(image.dim()[0], (self.d, self.N, 1))
        self.assertEqual(image.dim()[1], self.bsz)
        # Call on image
        image = util.matrix_to_image(image)
        # Call on 4d tensor'
        self.assertRaises(
            ValueError,
            util.matrix_to_image,
            dy.inputTensor([[[[0]]]])
        )

    def test_conditional_dropout(self):
        dy.renew_cg()
        x = dy.random_uniform(100, 0, 1)
        # Now test all cases
        self.assertFalse(np.allclose(
            util.conditional_dropout(x, self.dropout_rate, True).npvalue(),
            x.npvalue(),
        ))
        self.assertTrue(np.allclose(
            util.conditional_dropout(x, self.dropout_rate, False).npvalue(),
            x.npvalue()
        ))
        self.assertTrue(np.allclose(
            util.conditional_dropout(x, 0.0, True).npvalue(),
            x.npvalue()
        ))
        self.assertTrue(np.allclose(
            util.conditional_dropout(x, 0.0, False).npvalue(),
            x.npvalue()
        ))
