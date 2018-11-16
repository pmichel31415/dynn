#!/usr/bin/env python3

import unittest
from unittest import TestCase

import numpy as np

from dynn.data import batching, dictionary


class TestNumpyBatches(TestCase):

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
        return batching.NumpyBatches(
            data, labels, batch_size=self.batch_size, shuffle=shuffle
        )

    def _dummy_regression_iterator(self):
        # Create dummy data
        data = np.random.uniform(size=(self.data_size, self.input_dim))
        # Create targets
        labels = np.random.uniform(size=(self.data_size, self.output_dim))
        # Iterator
        return batching.NumpyBatches(
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
        batched_dataset[10:1:-2]
        batched_dataset[::-1]


class TestPaddedSequenceBatches(TestCase):

    def setUp(self):
        self.dic = dictionary.Dictionary(symbols=list("abcdefg"))
        self.max_samples = 3
        self.max_tokens = 20
        self.num_labels = 10
        self.data_size = self.max_tokens + 1
        self.output_dim = 5
        self.data = [
            np.random.randint(low=self.dic.nspecials,
                              high=len(self.dic), size=i)
            for i in range(1, self.data_size + 1)
        ]

    def _dummy_classification_iterator(
        self,
        shuffle=True,
        group_by_length=False
    ):
        # Create targets
        labels = np.random.randint(self.num_labels, size=self.data_size)
        # Iterator
        return batching.PaddedSequenceBatches(
            self.data,
            labels,
            self.dic,
            max_samples=self.max_samples,
            max_tokens=self.max_tokens,
            shuffle=shuffle,
            group_by_length=group_by_length,
        )

    def _dummy_regression_iterator(self):
        # Create targets
        targets = np.random.uniform(size=(self.data_size, self.output_dim))
        # Iterator
        return batching.PaddedSequenceBatches(
            self.data,
            targets,
            self.dic,
            max_samples=self.max_samples,
            max_tokens=self.max_tokens,
        )

    def test_classification(self):
        batched_dataset = self._dummy_classification_iterator()
        # Try iterating
        for x, y in batched_dataset:
            # check dimension
            self.assertEqual(x.sequences.shape[0], max(x.lengths))
            self.assertLessEqual(x.sequences.shape[1], self.max_samples)
            self.assertLessEqual(
                len([
                    w for seq in x.sequences
                    for w in seq
                    if w != batched_dataset.pad_idx
                ]),
                self.max_tokens
            )
            self.assertEqual(len(y.shape), 1)
            self.assertEqual(x.sequences.shape[1], y.shape[0])

    def test_regression(self):
        batched_dataset = self._dummy_regression_iterator()
        # Try iterating
        for x, y in batched_dataset:
            # check dimension
            self.assertEqual(x.sequences.shape[0], max(x.lengths))
            self.assertLessEqual(x.sequences.shape[1], self.max_samples)
            self.assertLessEqual(
                len([
                    w for seq in x.sequences
                    for w in seq
                    if w != batched_dataset.pad_idx
                ]),
                self.max_tokens
            )
            self.assertEqual(y.shape[0], self.output_dim)
            self.assertEqual(x.sequences.shape[1], y.shape[1])

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

    def test_original_order(self):
        batched_dataset = self._dummy_classification_iterator(shuffle=True)
        # Record the original labels
        labels = batched_dataset.targets
        # Labels in the prder that they are provided by the iterator
        batches = [(batch, y) for batch, y in batched_dataset]
        random_labels = np.concatenate([y for _, y in batches])
        # Pointer to the original index
        order = np.concatenate([batch.original_idxs for batch, _ in batches])
        # Check that the given order is legit
        self.assertTrue(np.allclose(labels[order], random_labels))

    def test_length(self):
        batched_dataset = self._dummy_classification_iterator()
        num_batches = len([x for x in batched_dataset])
        self.assertEqual(num_batches, len(batched_dataset))

    def test_getitem(self):
        batched_dataset = self._dummy_classification_iterator()
        batched_dataset[0]
        batched_dataset[1:3]
        batched_dataset[1:10:2]
        batched_dataset[10:1:-2]
        batched_dataset[::-1]


class TestBPTTBatches(TestCase):

    def setUp(self):
        self.dic = dictionary.Dictionary(symbols=list("abcdefg"))
        self.batch_size = 3
        self.seq_length = 20
        self.data_size = self.seq_length * self.batch_size + 1
        self.data = np.random.randint(
            low=self.dic.nspecials, high=len(self.dic), size=self.data_size
        )

    def _dummy_classification_iterator(self):
        # Iterator
        return batching.BPTTBatches(
            self.data,
            batch_size=self.batch_size,
            seq_length=self.seq_length,
        )

    def test_classification(self):
        batched_dataset = self._dummy_classification_iterator()
        # Try iterating
        for x, y in batched_dataset:
            # Check dimensions
            self.assertTupleEqual(x.shape, y.shape)
            self.assertLessEqual(x.shape[0], self.seq_length)
            self.assertEqual(x.shape[1], self.batch_size)
            # Check values
            self.assertTrue(np.allclose(y[:-1], x[1:]))

    def test_length(self):
        batched_dataset = self._dummy_classification_iterator()
        num_batches = len([x for x in batched_dataset])
        self.assertEqual(num_batches, len(batched_dataset))

    def test_getitem(self):
        batched_dataset = self._dummy_classification_iterator()
        batched_dataset[0]
        batched_dataset[1:3]
        batched_dataset[1:10:2]
        batched_dataset[10:1:-2]
        batched_dataset[::-1]


class TestSequencePairsBatches(TestCase):

    def setUp(self):
        self.src_dic = dictionary.Dictionary(symbols=list("abcdefg"))
        self.tgt_dic = dictionary.Dictionary(symbols=list("hijklmn"))
        self.max_samples = 6
        self.max_tokens = 40
        self.data_size = self.max_tokens//2 + 1
        max_size = self.data_size + 1
        self.src_data = [
            np.random.randint(low=self.src_dic.nspecials,
                              high=len(self.src_dic), size=(i+3) % max_size)
            for i in range(1, max_size)
        ]
        self.tgt_data = [
            np.random.randint(low=self.tgt_dic.nspecials,
                              high=len(self.tgt_dic), size=(i+5) % max_size)
            for i in range(1, self.data_size + 1)
        ]
        self.labels = np.random.randint(3, size=self.data_size)

    def _dummy_iterator(
        self,
        shuffle=True,
        group_by_length=None,
        labelled=False,
    ):
        # Iterator
        return batching.SequencePairsBatches(
            self.src_data,
            self.tgt_data,
            self.src_dic,
            self.tgt_dic,
            labels=self.labels if labelled else None,
            max_samples=self.max_samples,
            max_tokens=self.max_tokens,
            shuffle=shuffle,
            group_by_length=group_by_length,
        )

    def test_not_grouped(self):
        batched_dataset = self._dummy_iterator()
        # Try iterating
        for x, y in batched_dataset:
            # check dimensions
            self.assertEqual(x.sequences.shape[0], max(x.lengths))
            self.assertLessEqual(x.sequences.shape[1], self.max_samples)
            self.assertLessEqual(
                len([w for seq in x.sequences for w in seq
                     if w != batched_dataset.src_pad_idx]) +
                len([w for seq in y.sequences for w in seq
                     if w != batched_dataset.tgt_pad_idx]),
                self.max_tokens
            )
            self.assertEqual(x.sequences.shape[1], y.sequences.shape[1])

    def test_grouped(self):
        for grouped_by in ["source", "target"]:
            batched_dataset = self._dummy_iterator(group_by_length=grouped_by)
            # Try iterating
            for x, y in batched_dataset:
                # check dimensions
                self.assertEqual(x.sequences.shape[0], max(x.lengths))
                self.assertLessEqual(x.sequences.shape[1], self.max_samples)
                self.assertLessEqual(
                    len([w for seq in x.sequences for w in seq
                         if w != batched_dataset.src_pad_idx]) +
                    len([w for seq in y.sequences for w in seq
                         if w != batched_dataset.tgt_pad_idx]),
                    self.max_tokens
                )
                self.assertEqual(x.sequences.shape[1], y.sequences.shape[1])

    def test_labelled(self):
        batched_dataset = self._dummy_iterator(labelled=True)
        # Try iterating
        for x, y, label in batched_dataset:
            # check dimensions
            self.assertEqual(x.sequences.shape[0], max(x.lengths))
            self.assertLessEqual(x.sequences.shape[1], self.max_samples)
            self.assertLessEqual(
                len([w for seq in x.sequences for w in seq
                     if w != batched_dataset.src_pad_idx]) +
                len([w for seq in y.sequences for w in seq
                     if w != batched_dataset.tgt_pad_idx]),
                self.max_tokens
            )
            self.assertEqual(x.sequences.shape[1], y.sequences.shape[1])
            self.assertTupleEqual(label.shape, (x.sequences.shape[1],))

    def test_shuffle(self):
        batched_dataset = self._dummy_iterator(shuffle=True)
        # Record the source sequences for the first 2 epochs
        first_epoch_srcs = np.concatenate(
            [src.sequences.flatten()
             for src, _ in batched_dataset]
        )
        second_epoch_srcs = np.concatenate(
            [src.sequences.flatten()
             for src, _ in batched_dataset]
        )
        # Check that the sequences are not all equals (the probability that
        # this happens by chance and the test fails is very low)
        self.assertFalse(np.allclose(first_epoch_srcs, second_epoch_srcs))

    def test_no_shuffle(self):
        batched_dataset = self._dummy_iterator(shuffle=False)
        # Record the source sequences for the first 2 epochs
        first_epoch_srcs = np.concatenate(
            [src.sequences.flatten()
             for src, _ in batched_dataset]
        )
        second_epoch_srcs = np.concatenate(
            [src.sequences.flatten()
             for src, _ in batched_dataset]
        )
        # Check that the data is equal
        self.assertTrue(np.allclose(first_epoch_srcs, second_epoch_srcs))

    def test_original_order(self):
        # Now grouped by length
        batched_dataset = self._dummy_iterator(
            shuffle=True, group_by_length="source")
        # Labels in the prder that they are provided by the iterator
        batches = [batch for batch in batched_dataset]
        random_seqs = np.asarray([
            w for _, tgt in batches
            for y in tgt.sequences.T
            for w in y if w != self.tgt_dic.pad_idx]).astype(int)
        # Pointer to the original index
        order = np.asarray(
            [idx for _, tgt in batches for idx in tgt.original_idxs])
        # Check that the given order is legit
        ordered_seqs = np.asarray([
            w for idx in order
            for w in self.tgt_data[idx]
        ]).astype(int)
        self.assertListEqual(ordered_seqs.tolist(), random_seqs.tolist())

    def test_length(self):
        batched_dataset = self._dummy_iterator()
        num_batches = len([x for x in batched_dataset])
        self.assertEqual(num_batches, len(batched_dataset))

    def test_getitem(self):
        batched_dataset = self._dummy_iterator()
        batched_dataset[0]
        batched_dataset[1:3]
        batched_dataset[1:10:2]
        batched_dataset[10:1:-2]
        batched_dataset[::-1]


if __name__ == '__main__':
    unittest.main()
