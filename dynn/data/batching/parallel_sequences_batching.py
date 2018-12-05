#!/usr/bin/env python3
import logging

import numpy as np

from .sequence_batch import SequenceBatch


class SequencePairsBatches(object):
    """Wraps two lists of sequences as a batch iterator.

    This is useful for sequence-to-sequence problems or sentence pairs
    classification (entailment, paraphrase detection...). Following seq2seq
    conventions the first sequence is referred to as the "source" and the
    second as the "target".

    You can then iterate over this object and get tuples of
    ``src_batch, tgt_batch`` ready for use in your computation graph.

    Example:

    .. code-block:: python

        # Dictionary
        dic = dynn.data.dictionary.Dictionary(symbols="abcde".split())
        # 1000 source sequences of various lengths up to 10
        src_data = [np.random.randint(len(dic), size=np.random.randint(10))
                    for _ in range(1000)]
        # 1000 target sequences of various lengths up to 10
        tgt_data = [np.random.randint(len(dic), size=np.random.randint(10))
                    for _ in range(1000)]
        # Iterator with at most 20 samples or 50 tokens per batch
        batched_dataset = SequencePairsBatches(
            src_data, tgt_data, max_samples=20
        )
        # Training loop
        for x, y in batched_dataset:
            # x and y are SequenceBatch objects

    Args:
        src_data (list): List of source sequences (list of int iterables)
        tgt_data (list): List of target sequences (list of int iterables)
        src_dictionary (Dictionary): Source dictionary
        tgt_dictionary (Dictionary): Target dictionary
        max_samples (int, optional): Maximum number of samples per batch (one
            sample is a pair of sentences)
        max_tokens (int, optional): Maximum number of total tokens per batch
            (source + target tokens)
        strict_token_limit (bool, optional): Padding tokens will count towards
            the ``max_tokens`` limit
        shuffle (bool, optional): Shuffle the dataset whenever starting a new
            iteration (default: ``True``)
        group_by_length (str, optional): Group sequences by length. One of
            ``"source"`` or ``"target"``. This minimizes the number of padding
            tokens. The batches are not strictly IID though.
        src_left_aligned (bool, optional): Align the source sequences to the
            left
        tgt_left_aligned (bool, optional): Align the target sequences to the
            left
    """

    def __init__(
        self,
        src_data,
        tgt_data,
        src_dictionary,
        tgt_dictionary=None,
        labels=None,
        max_samples=32,
        max_tokens=99999999,
        strict_token_limit=False,
        shuffle=True,
        group_by_length="source",
        src_left_aligned=True,
        tgt_left_aligned=True,
    ):
        if len(src_data) != len(tgt_data):
            raise ValueError(
                f"Source and target data mismatch: "
                f"{len(src_data)} != {len(tgt_data)}"
            )
        self.num_samples = len(src_data)
        self.src_size = sum(len(src) for src in src_data)
        self.tgt_size = sum(len(tgt) for tgt in tgt_data)
        # Main parameters
        self.max_samples = max_samples
        self.max_tokens = max_tokens
        self.strict_token_limit = strict_token_limit
        self.shuffle = shuffle
        self.src_left_aligned = src_left_aligned
        self.tgt_left_aligned = tgt_left_aligned
        self.src_pad_idx = src_dictionary.pad_idx
        if tgt_dictionary is not None:
            self.tgt_pad_idx = tgt_dictionary.pad_idx
        else:
            self.tgt_pad_idx = self.src_pad_idx
        self.group_by_src_length = False
        self.group_by_tgt_length = False
        if group_by_length is "source":
            self.group_by_src_length = True
        elif group_by_length is "target":
            self.group_by_tgt_length = True
        elif group_by_length is not None:
            raise ValueError(
                "group_by_length bust be one of None, "
                "\"source\" or \"target\""
            )
        self.group_by_length = self.group_by_src_length or \
            self.group_by_tgt_length
        # Maybe order the data by lengths
        initial_order = np.arange(self.num_samples)
        if self.group_by_src_length:
            initial_order = np.argsort([len(x) for x in src_data])
        elif self.group_by_tgt_length:
            initial_order = np.argsort([len(x) for x in tgt_data])
        # Store the data in the appropriate order
        self.src_data = np.asarray([src_data[idx] for idx in initial_order])
        self.tgt_data = np.asarray([tgt_data[idx] for idx in initial_order])
        # Keep track of the original position of each sentence
        self.original_position = initial_order
        # Handle labels
        self.labelled = labels is not None
        if self.labelled:
            # Check size
            if len(labels) != self.num_samples:
                raise ValueError(
                    "Number of samples and labels differ: "
                    f"{self.num_samples} != {len(labels)}"
                )
            # Store in appropriate order
            self.labels = np.asarray([labels[idx] for idx in initial_order])
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

        The result is a tuple ``src_batch, tgt_batch`` where each is a
        ``batch_data`` is a :py:class:`SequenceBatch` object

        Args:
            index (int, slice): Index or slice

        Returns:
            tuple: ``src_batch, tgt_batch``
        """
        src_batch = SequenceBatch(
            self.src_data[index],
            original_idxs=self.original_position[index],
            pad_idx=self.src_pad_idx,
            left_aligned=self.src_left_aligned,
        )
        tgt_batch = SequenceBatch(
            self.tgt_data[index],
            original_idxs=self.original_position[index],
            pad_idx=self.tgt_pad_idx,
            left_aligned=self.tgt_left_aligned,
        )
        if self.labelled:
            return src_batch, tgt_batch, self.labels[index]
        else:
            return src_batch, tgt_batch

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
        n_tokens = n_samples = max_src_len = max_tgt_len = 0
        for idx in data_order:
            src_sample = self.src_data[idx]
            tgt_sample = self.tgt_data[idx]
            len_sample = len(src_sample) + len(tgt_sample)
            # If the sample itself causes the overflow, ignore it
            # with a warning
            if len_sample > self.max_tokens:
                logging.warning(f"Discarding one sample of size {len_sample}")
                continue
            # Check if there are too many tokens/samples
            too_many_samples = n_samples + 1 > self.max_samples
            if self.strict_token_limit:
                max_len = max(max_src_len, len(src_sample))
                max_len += max(max_tgt_len, len(tgt_sample))
                too_many_tokens = max_len * (n_samples + 1) > self.max_tokens
            else:
                too_many_tokens = n_tokens + len_sample > self.max_tokens
            # Handle the case if the batch is finished
            if too_many_samples or too_many_tokens:
                # Add current batch and start a new one
                batches.append(current_batch)
                current_batch = []
                n_tokens = n_samples = max_src_len = max_tgt_len = 0
            # Add the sample to the current batch
            current_batch.append(idx)
            n_samples += 1
            n_tokens += len_sample
            max_src_len = max(max_src_len, len(src_sample))
            max_tgt_len = max(max_tgt_len, len(tgt_sample))
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
