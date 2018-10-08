#!/usr/bin/env python3

import unittest
from unittest import TestCase

from dynn.data import dictionary


class TestDictionary(TestCase):

    def setUp(self):
        self.data = ["a a a a a b b c c c".split()]
        self.forced_symbols = ["c"]

    def test_from_data(self):
        dic = dictionary.Dictionary.from_data(self.data)
        expected_symbols = {
            dictionary.UNK_TOKEN,
            dictionary.PAD_TOKEN,
            dictionary.EOS_TOKEN,
            "a",
            "c",
            "b",
        }
        self.assertSetEqual(set(dic.symbols), expected_symbols)


if __name__ == '__main__':
    unittest.main()
