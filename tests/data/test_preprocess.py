#!/usr/bin/env python3

import unittest
from unittest import TestCase

import numpy as np

from dynn.data import preprocess


class TestPreprocess(TestCase):

    def test_normalize_array(self):
        x = np.random.normal(loc=3.14, scale=1.41, size=(10, 20))
        x_norm = preprocess.normalize(x)
        self.assertAlmostEqual(x_norm.mean(), 0.0, 10)
        self.assertAlmostEqual(x_norm.std(), 1.0, 10)

    def test_normalize_array_list(self):
        x = [np.random.normal(loc=3.14, scale=1.41, size=(10, 20))
             for _ in range(5)]
        x_norm = preprocess.normalize(x)
        for elem in x_norm:
            self.assertAlmostEqual(elem.mean(), 0.0, 10)
            self.assertAlmostEqual(elem.std(), 1.0, 10)

    def test_lowercase_string(self):
        x = "abcdABCD"
        x_low = preprocess.lowercase(x)
        self.assertEqual(x_low, "abcdabcd")

    def test_lowercase_string_list(self):
        x = ["abcdABCD" for _ in range(5)]
        x_low = preprocess.lowercase(x)
        for elem in x_low:
            self.assertEqual(elem, "abcdabcd")


if __name__ == '__main__':
    unittest.main()
