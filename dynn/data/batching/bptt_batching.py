#!/usr/bin/env python3

import numpy as np


class BPTTBatches(object):
    """Wraps a list of sequences as a contiguous batch iterator.

    This will iterate over batches of contiguous subsequences of size
    ``seq_length``. TODO: elaborate

    Example:

    .. code-block:: python

        # Dictionary
        # Sequence of length 1000
        data = np.random.randint(10, size=1000)
        # Iterator with over subsequences of length 20 with batch size 5
        batched_dataset = BPTTBatches(data, batch_size=5, seq_length=20)
        # Training loop
        for x, y in batched_dataset:
            # x has and y have shape (seq_length, batch_size)
            # y[i+1] == x[i]
            # Do something with x

    Args:
        data (list): List of numpy arrays containing the data
        targets (list): List of targets
        batch_size (int, optional): Batch size
        seq_length (int, optional): BPTT length
    """

    def __init__(
        self,
        data,
        batch_size=32,
        seq_length=30,
    ):
        # Get one list
        if isinstance(data[0], list):
            data = [word for sent in data for word in sent]
        # Parameters
        self.num_samples = len(data)
        self.num_samples -= self.num_samples % batch_size
        self.num_positions = self.num_samples//batch_size
        self.num_batches = int(np.ceil(self.num_positions / seq_length))
        self.batch_size = batch_size
        self.seq_length = seq_length
        # The data is stored as an array of shape (-1, batch_size)
        self.data = np.stack([
            np.asarray(
                data[b*self.num_positions:(b+1)*self.num_positions],
                dtype=type(data[0])
            )
            for b in range(self.batch_size)],
            axis=-1
        )
        # Reset position and shuffle the order if applicable
        self.reset()

    def __len__(self):
        """This returns the number of **batches** in the dataset
        (not the total number of samples)

        Returns:
            int: Number of batches in the dataset
                ``ceil(len(data)/batch_size)``
        """
        return self.num_batches

    def __getitem__(self, index):
        """Returns the ``index`` th sample

        The result is a tuple ``x, next_x`` of numpy arrays of shape
        ``seq_len x batch_size`` ``seq_length`` is determined by the range
        specified by ``index``, and ``next_x[t]=x[t+1]`` for all ``t``

        Args:
            index (int, slice): Index or slice

        Returns:
            tuple: ``x, next_x``
        """
        return self.data[index]

    def percentage_done(self):
        """What percent of the data has been covered in the current epoch"""
        return 100 * (self.position / self.num_positions)

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
        return (self.position // self.seq_length) % batch_number == 0

    def reset(self):
        """Reset the iterator and shuffle the dataset if applicable"""
        self.position = 0

    def __iter__(self):
        self.reset()
        return self

    def __next__(self):
        # Check for end of epoch
        if self.position >= self.num_positions-1:
            raise StopIteration
        # Batch index
        seq_len = min(self.seq_length, self.num_positions-1-self.position)
        batch = self[self.position:self.position+seq_len+1]
        # Increment position
        self.position += seq_len
        # Return batch
        return batch[:-1], batch[1:]
