#!/usr/bin/env python3

import unittest
from unittest import TestCase

import sys
import argparse

from dynn import command_line


class TestCommandLine(TestCase):

    def setUp(self):
        self.actual_sysargs = sys.argv[:]
        sys.argv = [
            sys.argv[0],
            "--dynet-seed", "42",
            "--dynet-gpu",
            "--dynet-gpus", "1",
            "--dynet-devices", "CPU",
            "--dynet-mem", "23",
            "--dynet-autobatch", "1",
            "--dynet-profiling", "1",
            "--dynet-weight-decay", "0.1",
        ]
        self.parser = argparse.ArgumentParser()

    def tearDown(self):
        sys.argv = self.actual_sysargs[:]

    def test_dont_add_dynet_args(self):

        self.assertRaises(SystemExit, self.parser.parse_args)

    def test_add_dynet_args(self):
        command_line.add_dynet_args(self.parser)

        args = self.parser.parse_args()
        self.assertTrue(hasattr(args, "dynet_seed"))
        self.assertTrue(hasattr(args, "dynet_gpu"))
        self.assertTrue(hasattr(args, "dynet_gpus"))
        self.assertTrue(hasattr(args, "dynet_devices"))
        self.assertTrue(hasattr(args, "dynet_mem"))
        self.assertTrue(hasattr(args, "dynet_autobatch"))
        self.assertTrue(hasattr(args, "dynet_profiling"))
        self.assertTrue(hasattr(args, "dynet_weight_decay"))


if __name__ == '__main__':
    unittest.main()
