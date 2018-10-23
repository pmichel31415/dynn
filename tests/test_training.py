#!/usr/bin/env python3

import unittest
from unittest import TestCase

import numpy as np

from dynn import training


class TestTraining(TestCase):

    def setUp(self):
        self.warmup = 6
        self.max_steps = 100
        self.lr0 = 42.0

    def test_inverse_sqrt_schedule(self):
        learning_rate = training.inverse_sqrt_schedule(self.warmup, self.lr0)
        # First max_steps learning rates
        lrs = [lr for _, lr in zip(range(self.max_steps), learning_rate)]
        # Check at a few key values
        self.assertAlmostEqual(lrs[0], 0.0)
        self.assertAlmostEqual(lrs[self.warmup // 2],
                               self.lr0 / (2 * np.sqrt(self.warmup)))
        self.assertAlmostEqual(lrs[self.warmup],
                               self.lr0 / np.sqrt(self.warmup))


if __name__ == '__main__':
    unittest.main()
