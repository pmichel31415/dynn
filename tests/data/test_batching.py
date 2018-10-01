#!/usr/bin/env python3

import unittest
from unittest import TestCase

import numpy as np

from dynn.data import batching


class TestNumpyBatchIterator(TestCase):

    def setUp(self):
        self.input_dim = 3
        self.output_dim = 2
        self.num_labels = 50
        self.batch_size = 5
        self.data_size = 101

    def _dummy_classification_iterator(self, shuffle=True):
        # Create dummy data
        data = np.random.uniform(size=(self.data_size, self.input_dim))
        # Create targets
        labels = np.random.randint(self.num_labels, size=self.data_size)
        # Iterator
        return batching.NumpyBatchIterator(
            data, labels, batch_size=self.batch_size, shuffle=shuffle
        )

    def _dummy_regression_iterator(self):
        # Create dummy data
        data = np.random.uniform(size=(self.data_size, self.input_dim))
        # Create targets
        labels = np.random.uniform(size=(self.data_size, self.output_dim))
        # Iterator
        return batching.NumpyBatchIterator(
            data, labels, batch_size=self.batch_size
        )

    def test_classification(self):
        batched_dataset = self._dummy_classification_iterator()
        # Try iterating
        for x, y in batched_dataset:
            self.assertEqual(x.shape[0], (self.input_dim))
            self.assertIn(
                x.shape[1],
                {self.batch_size, self.data_size % self.batch_size}
            )
            self.assertEqual(len(y.shape), 1)
            self.assertEqual(x.shape[1], y.shape[0])

    def test_regression(self):
        batched_dataset = self._dummy_regression_iterator()
        # Try iterating
        for x, y in batched_dataset:
            self.assertEqual(x.shape[0], self.input_dim)
            self.assertIn(
                x.shape[1],
                {self.batch_size, self.data_size % self.batch_size}
            )
            self.assertEqual(y.shape[0], self.output_dim)
            self.assertEqual(x.shape[1], y.shape[1])

    def test_shuffle(self):
        batched_dataset = self._dummy_classification_iterator(shuffle=True)
        # Record the labels for the first 2 epochs
        first_epoch_labels = np.concatenate([y for _, y in batched_dataset])
        second_epoch_labels = np.concatenate([y for _, y in batched_dataset])
        # Check that the labels are not all equals (the probability that this
        # happens by chance and the test fails is very low given the number of
        # labels)
        self.assertFalse(np.allclose(first_epoch_labels, second_epoch_labels))

    def test_no_shuffle(self):
        batched_dataset = self._dummy_classification_iterator(shuffle=False)
        # Record the labels for the first 2 epochs
        first_epoch_labels = np.concatenate([y for _, y in batched_dataset])
        second_epoch_labels = np.concatenate([y for _, y in batched_dataset])
        # Check that the labels are not all equals (the probability that this
        # happens by chance and the test fails is very low given the number of
        # labels)
        self.assertTrue(np.allclose(first_epoch_labels, second_epoch_labels))

    def test_length(self):
        batched_dataset = self._dummy_classification_iterator()
        num_batches = len([x for x in batched_dataset])
        self.assertEqual(num_batches, len(batched_dataset))

    def test_getitem(self):
        batched_dataset = self._dummy_classification_iterator()
        batched_dataset[0]
        batched_dataset[1:3]
        batched_dataset[1:10:2]
        batched_dataset[1:10:-2]
        batched_dataset[::-1]


if __name__ == '__main__':
    unittest.main()
