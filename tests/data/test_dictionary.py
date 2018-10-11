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

    def test_numberize(self):
        dic = dictionary.Dictionary.from_data(self.data)
        numberized_data = dic.numberize(self.data[0])
        numberized_data.append(dic.pad_idx)
        # Stringify without padding
        string = dic.string(numberized_data, join_with=" ")
        self.assertEqual(string, " ".join(self.data[0]))
        # Stringify with padding
        string = dic.string(numberized_data, with_pad=True, join_with=" ")
        padded_data = self.data[0] + [dic[dic.pad_idx]]
        self.assertEqual(string, " ".join(padded_data))


if __name__ == '__main__':
    unittest.main()
