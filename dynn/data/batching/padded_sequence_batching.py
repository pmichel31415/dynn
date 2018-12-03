#!/usr/bin/env python3
import logging

import numpy as np

from .sequence_batch import SequenceBatch


class PaddedSequenceBatches(object):
    """Wraps a list of sequences and a list of targets as a batch iterator.

    You can then iterate over this object and get tuples of
    ``batch_data, batch_targets`` ready for use in your computation graph.

    Example:

    .. code-block:: python

        # Dictionary
        dic = dynn.data.dictionary.Dictionary(symbols="abcde".split())
        # 1000 sequences of various lengths up to 10
        data = [np.random.randint(len(dic), size=np.random.randint(10))
                for _ in range(1000)]
        # Class labels
        labels = np.random.randint(10, size=1000)
        # Iterator with at most 20 samples or 50 tokens per batch
        batched_dataset = PaddedSequenceBatches(
            data,
            targets=labels,
            max_samples=20,
            pad_idx=dic.pad_idx,
        )
        # Training loop
        for x, y in batched_dataset:
            # x is a SequenceBatch object
            # and y has shape (batch_size,)
            # Do something with x and y

        # Without labels
        batched_dataset = PaddedSequenceBatches(
            data,
            max_samples=20,
            pad_idx=dic.pad_idx,
        )
        for x in batched_dataset:
            # x is a SequenceBatch object
            # Do something with x


    Args:
        data (list): List of numpy arrays containing the data
        targets (list): List of targets
        pad_value (number): Value at padded position
        max_samples (int, optional): Maximum number of samples per batch
        max_tokens (int, optional): Maximum number of tokens per batch. This
            count doesn't include padding tokens
        shuffle (bool, optional): Shuffle the dataset whenever starting a new
            iteration (default: ``True``)
        group_by_length (bool, optional): Group sequences by length. This
            minimizes the number of padding tokens. The batches are not
            strictly IID though.
        left_aligned (bool, optional): Align the sequences to the left
    """

    def __init__(
        self,
        data,
        targets=None,
        max_samples=32,
        pad_idx=0,
        max_tokens=np.inf,
        shuffle=True,
        group_by_length=True,
        left_aligned=True,
    ):
        self.labelled = targets is not None
        if self.labelled and len(data) != len(targets):
            raise ValueError(
                f"Data and targets size mismatch ({len(data)} "
                f"vs {len(targets)})"
            )
        self.num_samples = len(data)
        # The data is stored as an array of lists sorted by length
        self.data = np.asarray(
            sorted(data, key=lambda x: len(x)), dtype=object
        )
        if self.labelled:
            self.targets = np.asfortranarray(
                np.stack([target for target in targets], axis=-1)
            )
        # Main parameters
        self.max_samples = max_samples
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.group_by_length = group_by_length
        self.left_aligned = left_aligned
        self.pad_idx = pad_idx
        # Initial position and order
        self.position = 0
        self.batches = []
        # Keep track of each sequence's position
        self.original_position = np.arange(self.num_samples, dtype=int)
        # Reset position and shuffle the order if applicable
        self.reset()

    def __len__(self):
        """This returns the number of **batches** in the dataset
        (not the total number of samples)

        Returns:
            int: Number of batches in the dataset
                ``ceil(len(data)/batch_size)``
        """
        return len(self.batches)

    def __getitem__(self, index):
        """Returns the ``index`` th sample

        The result is a tuple ``batch_data, batch_target`` where the first is
        a batch of sequences and the other is is a numpy array in Fortran
        layout (for more efficient input in dynet).

        ``batch_data`` is a :py:class:`SequenceBatch` object

        Args:
            index (int, slice): Index or slice

        Returns:
            tuple: ``batch_data, batch_target``
        """
        batch_data = SequenceBatch(
            self.data[index],
            original_idxs=self.original_position[index],
            pad_idx=self.pad_idx,
            left_aligned=self.left_aligned,
        )
        if not self.labelled:
            return batch_data
        else:
            batch_targets = self.targets[..., index]
            return batch_data, batch_targets

    def percentage_done(self):
        """What percent of the data has been covered in the current epoch"""
        return 100 * (self.position / len(self.batches))

    def just_passed_multiple(self, batch_number):
        """Checks whether the current number of batches processed has
        just passed a multiple of ``batch_number``.

        For example you can use this to report at regular interval
        (eg. every 10 batches)

        Args:
            batch_number (int): [description]

        Returns:
            bool: ``True`` if :math:`\\fraccurrent_batch`
        """
        return self.position % batch_number == 0

    def reset(self):
        """Reset the iterator and shuffle the dataset if applicable"""
        self.position = 0
        # If the sentences aren't grouped by length, shuffle them now
        data_order = np.arange(self.num_samples)
        if self.shuffle and not self.group_by_length:
            np.random.shuffle(data_order)
        # Group the sentence into batches with appropriate size
        batches = []
        current_batch = []
        n_tokens = n_samples = 0
        for idx in data_order:
            sample = self.data[idx]
            # Handle the case if the batch is finished
            if len(sample) > self.max_tokens:
                logging.warning(f"Discarding one sample of size {len(sample)}")
                continue
            # Check if there are too many tokens/samples
            too_many_samples = n_samples + 1 > self.max_samples
            too_many_tokens = n_tokens + len(sample) > self.max_tokens
            # Handle the case if the batch is finished
            if too_many_samples or too_many_tokens:
                # Add current batch and start a new one
                batches.append(current_batch)
                current_batch = []
                n_tokens = n_samples = 0
            # Add the sample to the current batch
            current_batch.append(idx)
            n_samples += 1
            n_tokens += len(sample)
        # Add last batch
        if len(current_batch) != 0:
            batches.append(current_batch)
        # Batches
        self.batches = np.asarray(batches)
        # Shuffle again (but this time the batches)
        if self.shuffle:
            np.random.shuffle(self.batches)

    def __iter__(self):
        self.reset()
        return self

    def __next__(self):
        # Check for end of epoch
        if self.position >= len(self.batches):
            raise StopIteration
        # Batch index
        indices = self.batches[self.position]
        # Increment position
        self.position += 1
        # Return batch
        return self[indices]
