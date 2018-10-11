#!/usr/bin/env python3
import unittest
from unittest import TestCase

import numpy as np
import dynet as dy

from dynn.data.dictionary import Dictionary
from dynn.layers import embedding_layers


class TestEmbeddings(TestCase):

    def setUp(self):
        self.pc = dy.ParameterCollection()
        self.dim = 10
        self.dic = Dictionary(symbols="abcdefg".split())
        self.bsz = 7

    def test_forward_backward(self):
        # Create compact lstm
        embed = embedding_layers.Embeddings(
            self.pc, self.dic, self.dim,
        )
        # Initialize computation graph
        dy.renew_cg()
        # Create inputs
        idxs = np.random.randint(len(self.dic), size=self.bsz)
        # Initialize layer
        embed.init(test=False, update=True)
        # Embed indices
        y = embed(idxs)
        # Try forward/backward
        z = dy.sum_batches(dy.sum_elems(y))
        z.forward()
        z.backward()
        # Test dim
        self.assertTupleEqual(y.dim()[0], (self.dim,))
        self.assertEqual(y.dim()[1], self.bsz)
        # Check values
        expected_values = embed.params.as_array()[idxs].transpose()
        self.assertTrue(np.allclose(y.npvalue(), expected_values))


if __name__ == '__main__':
    unittest.main()
