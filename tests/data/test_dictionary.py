#!/usr/bin/env python3

import unittest
from unittest import TestCase

import os.path
import tempfile
import shutil

from dynn.data import dictionary


class TestDictionary(TestCase):

    def setUp(self):
        self.data = ["a a a a a b b c c c".split()]
        self.forced_symbols = ["c"]
        self.path = tempfile.mkdtemp()
        self.filename = os.path.join(self.path, "test.dic")

    def tearDown(self):
        shutil.rmtree(self.path)

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

    def test_io(self):
        dic = dictionary.Dictionary.from_data(self.data, special_symbols=["z"])
        dic.thaw()
        print(dic.frozen)
        # Save to file
        dic.save(self.filename)
        # New dictionary
        dic1 = dictionary.Dictionary.load(self.filename)
        # Check equality
        self.assertListEqual(dic1.symbols, dic.symbols)
        self.assertFalse(dic1.frozen)
        self.assertEqual(dic1.nspecials, 4)
        # Save symbols only to file
        dic.save(self.filename, symbols_only=True)
        # New dictionary
        dic2 = dictionary.Dictionary.load(self.filename)
        # Check equality
        self.assertListEqual(dic2.symbols, dic.symbols)
        self.assertTrue(dic2.frozen)
        self.assertEqual(dic2.nspecials, 3)


if __name__ == '__main__':
    unittest.main()
