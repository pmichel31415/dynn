#!/usr/bin/env python3
from collections import Iterable

import numpy as np

from ...operations import seq_mask
from ...util import _default_value


class SequenceBatch(object):
    """Batched sequence object with padding

    This wraps a list of integer sequences into a nice array padded to the
    longest sequence. The batch dimension (number of sequences) is the last
    dimension.

    By default the sequences are padded to the right which means that they
    are aligned to the left (they all start at index 0)

    Args:
        sequences (list): List of list of integers
        original_idxs (list): This list should point to the original position
            of each sequence in the data (before shuffling/reordering). This is
            useful when you want to access information that has been discarded
            during preprocessing (eg original sentence before numberizing and
            ``<unk>`` ing in MT).
        pad_idx (int): Default index for padding
        left_aligned (bool, optional): Align to the left (all sequences start
            at the same position).
    """

    def __init__(
        self,
        sequences,
        original_idxs=None,
        pad_idx=None,
        left_aligned=True
    ):
        if len(sequences) == 0:
            raise ValueError("Can't batch 0 sequences together")
        if not isinstance(sequences[0], Iterable):
            sequences = [sequences]
        self.original_idxs = _default_value(original_idxs, [0]*len(sequences))
        self.lengths = [len(seq) for seq in sequences]
        self.pad_idx = _default_value(pad_idx, 0)
        self.left_aligned = left_aligned
        self.unpadded_sequences = sequences
        self.sequences = self.collate(sequences)
        self.max_length = self.sequences.shape[0]
        self.batch_size = self.sequences.shape[1]

    def __getitem__(self, index):
        return SequenceBatch(
            self.unpadded_sequences[index],
            self.original_idxs[index],
            self.pad_idx,
            self.left_aligned,
        )

    def get_mask(self, base_val=1, mask_val=0):
        """Return a mask expression with specific values for padding tokens.

        This will return an expression of the same shape as ``self.sequences``
        where the ``i`` th element of batch ``b`` is ``base_val`` iff
        ``i<=lengths[b]`` (and ``mask_val`` otherwise).

        For example, if ``size`` is ``4`` and ``lengths`` is ``[1,2,4]`` then
        the returned mask will be:

        .. code-block::

            1 0 0 0
            1 1 0 0
            1 1 1 1

        (here each row is a batch element)

        Args:
            base_val (int, optional): Value of the mask for non-masked indices
                (typically 1 for multiplicative masks and 0 for additive
                masks). Defaults to 1.
            mask_val (int, optional): Value of the mask for masked indices
                (typically 0 for multiplicative masks and -inf for additive
                masks). Defaults to 0.

        """
        mask = seq_mask(
            self.max_length,
            self.lengths,
            base_val=base_val,
            mask_val=mask_val,
            left_aligned=self.left_aligned
        )
        return mask

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
            if self.left_aligned:
                batch_array[:self.lengths[batch_idx], batch_idx] = sequence
            else:
                batch_array[-self.lengths[batch_idx]:, batch_idx] = sequence
        return batch_array
