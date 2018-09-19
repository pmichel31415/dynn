#!/usr/bin/env python3

import unittest
from unittest import TestCase
import tempfile
import shutil

from dynn.data import data_util


class TestDataUtil(TestCase):

    def setUp(self):
        self.file = "README.md"
        self.url = "https://raw.githubusercontent.com/pmichel31415/dynn/master/"  # noqa
        self.path = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.path)

    def test_download_if_not_here(self):
        # Initial download
        self.assertIsNotNone(data_util.download_if_not_there(
            self.file, self.url, self.path, force=False)
        )
        # Retry (shouldn't download)
        self.assertIsNone(data_util.download_if_not_there(
            self.file, self.url, self.path, force=False)
        )
        # Retry with force=True (should re-download)
        self.assertIsNotNone(data_util.download_if_not_there(
            self.file, self.url, self.path, force=True)
        )


if __name__ == '__main__':
    unittest.main()
