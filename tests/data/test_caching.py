#!/usr/bin/env python3

import unittest
from unittest import TestCase

import os
import tempfile
import shutil
import numpy as np

from dynn.data import caching


class TestCaching(TestCase):

    def setUp(self):
        self.path = tempfile.mkdtemp()
        rand_idx = np.random.randint(1e6)
        self.filename = os.path.join(self.path, f"test.{rand_idx}.bin")

    def tearDown(self):
        shutil.rmtree(self.path)

    def test_cached_to_file(self):
        # Dummy function

        @caching.cached_to_file(self.filename)
        def _dummy_function(x):
            return x + 1

        # Firts call
        two = _dummy_function(1)
        # Sanity check
        self.assertEqual(two, 2)

        # Second call
        also_two = _dummy_function(1)
        # Sanity check
        self.assertEqual(also_two, 2)

        # Call with different arguments
        maybe_three = _dummy_function(2)
        # Sanity check
        self.assertEqual(maybe_three, 2)

        # Call with different argument AND update cache
        three = _dummy_function(2, update_cache=True)
        # Sanity check
        self.assertEqual(three, 3)

        # Call with same argument but still update cache
        also_three = _dummy_function(2, update_cache=True)
        # Sanity check
        self.assertEqual(also_three, 3)


if __name__ == '__main__':
    unittest.main()
