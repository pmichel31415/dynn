#!/usr/bin/env python3
import logging
from collections import Iterable

import numpy as np


class BatchedSequence(object):
    """Batched sequence object with padding

    This wraps a list of integer sequences into a nice array padded to the
    longest sequence. The batch dimension (number of sequences) is the last
    dimension.

    By default the sequences are padded to the right which means that they
    are aligned to the left (they all start at index 0)

    Args:
        sequences (list): List of list of integers
        pad_idx (int): Default index for padding
        left_padded (bool, optional): Pad to the left (all sequences end at
            the same position).
    """

    def __init__(self, sequences, pad_idx, left_padded=False):
        if len(sequences) == 0:
            raise ValueError("Can't batch 0 sequences together")
        if not isinstance(sequences[0], Iterable):
            sequences = [sequences]
        self.lengths = [len(seq) for seq in sequences]
        self.pad_idx = pad_idx
        self.left_padded = False
        self.sequences = self.collate(sequences)

    def collate(self, sequences):
        """Pad and concatenate sequences to an array

        Args:
        sequences (list): List of list of integers
        pad_idx (int): Default index for padding

        Returns:
            :py:class:`np.ndarray`: ``max_len x batch_size`` array
        """
        max_len = max(self.lengths)
        # Initialize the array with the padding index
        batch_array = np.full(
            (max_len, len(sequences)), self.pad_idx, dtype=int
        )
        # Fill the indices values
        for batch_idx, sequence in enumerate(sequences):
            if self.left_padded:
                batch_array[-self.lengths[batch_idx]:, batch_idx] = sequence
            else:
                batch_array[:self.lengths[batch_idx], batch_idx] = sequence
        return batch_array


class PaddedSequenceBatchIterator(object):
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
        batched_dataset = PaddedSequenceBatchIterator(
            data, labels, max_samples=20
        )
        # Training loop
        for x, y in batched_dataset:
            # x has variable shape (length, batch_size)
            # and y has shape (batch_size,)
            # Do something with x and y

    Args:
        data (list): List of numpy arrays containing the data
        targets (list): List of targets
        dictionary ([type]): [description]
        max_samples (int, optional): Maximum number of samples per batch
        max_tokens (int, optional): Maximum number of tokens per batch. This
            count doesn't include padding tokens
        shuffle (bool, optional): Shuffle the dataset whenever starting a new
            iteration (default: ``True``)
        group_by_length (bool, optional): Group sequences by length. This
            minimizes the number of padding tokens. The batches are not
            strictly IID though.
        left_padded (bool, optional): Pad the sequences to the left
            (right-aligned batches)
    """

    def __init__(
        self,
        data,
        targets,
        dictionary,
        max_samples=32,
        max_tokens=np.inf,
        shuffle=True,
        group_by_length=True,
        left_padded=False,
    ):
        if len(data) != len(targets):
            raise ValueError(
                f"Data and targets size mismatch ({len(data)} "
                f"vs {len(targets)})"
            )
        self.num_samples = len(targets)
        # The data is stored as an array of lists sorted by length
        self.data = np.asarray(
            sorted(data, key=lambda x: len(x)), dtype=object
        )
        self.targets = np.asfortranarray(
            np.stack([target for target in targets], axis=-1)
        )
        # Main parameters
        self.max_samples = max_samples
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.group_by_length = group_by_length
        self.left_padded = left_padded
        self.pad_idx = dictionary.pad_idx
        # Initial position and order
        self.position = 0
        self.order = []
        self.batches = []
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

        ``batch_data`` is a :py:class:`BatchedSequence` object

        Args:
            index (int, slice): Index or slice

        Returns:
            tuple: ``batch_data, batch_target``
        """
        batch_data = BatchedSequence(
            self.data[index],
            pad_idx=self.pad_idx,
            left_padded=self.left_padded,
        )
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
            np.random.shuffle(self.order)
        # Group the sentence into batches with appropriate size
        batches = []
        current_batch = []
        n_tokens = n_samples = 0
        for idx in data_order:
            sample = self.data[idx]
            # Check if there are too many tokens/samples
            too_many_samples = n_samples + 1 > self.max_samples
            too_many_tokens = n_tokens + len(sample) > self.max_tokens
            # Handle the case if the batch is finished
            if too_many_samples or too_many_tokens:
                n_tokens = n_samples = 0
                if len(current_batch) == 0:
                    # If the sample itself causes the overflow, ignore it
                    # with a warning
                    logging.warning(
                        f"Discarding one sample of size {len(sample)}"
                    )
                    continue
                else:
                    # Add current batch and start a new one
                    batches.append(current_batch)
                    current_batch = []
            # Add the sample to the current batch
            if len(sample) > self.max_tokens:
                logging.warning(
                    f"Discarding one sample of size {len(sample)}"
                )
                continue
            else:
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
