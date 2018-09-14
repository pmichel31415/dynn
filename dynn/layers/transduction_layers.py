#!/usr/bin/env python3
"""
Sequence transduction layers
============================

Sequence transduction layers take in a sequence of expression and transduces
them using recurrent layers
"""

from ..util import _generate_mask, _should_mask, mask_batches
from .base_layers import BaseLayer


class UnidirectionalLayer(BaseLayer):
    """Unidirectional transduction layer

    This layer runs a recurrent cell on a sequence of inputs and produces
    resulting the sequence of recurrent states.

    Example:

    .. code-block:: python

        # LSTM cell
        lstm_cell = dynn.layers.LSTM(dy.ParameterCollection(), 10, 10)
        # Transduction layer
        lstm = dynn.layers.UnidirectionalLayer(lstm_cell)
        # Inputs
        dy.renew_cg()
        xs = [dy.random_uniform(10, batch_size=5) for _ in range(20)]
        # Initialize layer
        lstm.init(test=False)
        # Transduce forward
        states = lstm(xs)
        # Retrieve last h
        h_final = states[-1][0]

    Args:
        cell (:py:class:`recurrent_layers.RecurrentCell`): The
            recurrent cell to use for transduction
    """

    def __init__(self, cell):

        self.cell = cell

    def init(self, *args, **kwargs):
        """Passes its arguments to the recurrent layer"""
        self.cell.init(*args, **kwargs)

    def __call__(
        self,
        input_sequence,
        backward=False,
        lengths=None,
        left_padded=True
    ):
        """Transduces the sequence using the recurrent cell.

        The output is a list of the output states at each step.
        For instance in an LSTM the output is ``(h1, c1), (h2, c2), ...``

        This assumes that all the input expression have the same batch size.
        If you batch sentences of the same length together you should pad to
        the longest sequence.

        Args:
            input_sequence (list): Input as a list of
                :py:class:`dynet.Expression` objects
            backward (bool, optional): If this is ``True`` the sequence will
                be processed from left to right. The output sequence will
                still be in the same order as the input sequence though.
            lengths (list, optional): If the expressions in the sequence are
                batched, but have different lengths, this should contain a list
                of the sequence lengths (default: ``None``)
            left_padded (bool, optional): If the input sequences have different
                lengths they must be padded to the length of longest sequence.
                Use this to specify whether the sequence is left or right
                padded.

        Returns:
            list: List of recurrent states (depends on the recurrent layer)
        """
        # Dimensions of the input sequence
        batch_size = input_sequence[0].dim()[1]
        # Reverse the sequence for backward transduction
        max_length = len(input_sequence)
        if backward:
            input_sequence = reversed(input_sequence)
            # Padding is also reversed
            left_padded = not left_padded
        # Masking
        # TODO: separate function to make this code reusable as an iterator
        if lengths is None:
            # If we're not provided with a `length` argument no need
            # to do masking
            def do_masking(step): return False
        else:
            # This function will decide whether we need to apply a mask or not.
            # In particular we don't want to do masking if the mask is going to
            # be all ones
            def do_masking(step):
                return _should_mask(
                    step, min(lengths), max_length, left_padded
                )
        # Initial recurrent state provided by the recurrent cell
        state = self.cell.initial_value(batch_size=batch_size)
        # Start transducing
        output_sequence = []
        for step, x in enumerate(input_sequence):
            # Call the recurrent cell
            # The arguments to the recurrent layer will always be
            # recurrent state first and then input
            state = self.cell(x, *state)
            # Perform masking if we need to
            # TODO: masking -> interpolation with initial state
            if do_masking(step+1):
                # Generate the max depending on the position, padding, etc...
                mask = _generate_mask(
                    step+1, max_length, batch_size, lengths, left_padded
                )
                # Apply it
                state = mask_batches(state, mask, mode="mul")
            # Add the masked state to the output
            output_sequence.append(state)

        return output_sequence


class BidirectionalLayer(BaseLayer):
    """Bidirectional transduction layer

    This layer runs a recurrent cell on in each direction on a sequence of
    inputs and produces resulting the sequence of recurrent states.

    Example:

    .. code-block:: python

        # Parameter collection
        pc = dy.ParameterCollection()
        # LSTM cell
        fwd_lstm_cell = dynn.layers.LSTM(pc, 10, 10)
        bwd_lstm_cell = dynn.layers.LSTM(pc, 10, 10)
        # Transduction layer
        bilstm = dynn.layers.BidirectionalLayer(fwd_lstm_cell, bwd_lstm_cell)
        # Inputs
        dy.renew_cg()
        xs = [dy.random_uniform(10, batch_size=5) for _ in range(20)]
        # Initialize layer
        bilstm.init(test=False)
        # Transduce forward
        fwd_states, bwd_states = bilstm(xs)
        # Retrieve last h
        fwd_h_final = fwd_states[-1][0]
        # For the backward LSTM the final state is at
        # the beginning of the sequence (assuming left padding)
        bwd_h_final = fwd_states[0][0]

    Args:
        forward_cell (:py:class:`recurrent_layers.RecurrentCell`):The
            recurrent cell to use for forward transduction
        backward_cell (:py:class:`recurrent_layers.RecurrentCell`): The
            recurrent cell to use for backward transduction
    """

    def __init__(self, forward_cell, backward_cell):
        self.forward_transductor = UnidirectionalLayer(forward_cell)
        self.backward_transductor = UnidirectionalLayer(backward_cell)

    def init(self, *args, **kwargs):
        """Passes its arguments to the recurrent layers"""
        self.forward_transductor.init(*args, **kwargs)
        self.backward_transductor.init(*args, **kwargs)

    def __call__(
        self,
        input_sequence,
        lengths=None,
        left_padded=True
    ):
        """Transduces the sequence in both directions

        The output is a tuple ``forward_states, backward_states``  where each
        ``forward_states`` is a list of the output states of the forward
        recurrent cell at each step (and ``backward_states`` for the backward
        cell). For instance in a BiLSTM the output is
        ``[(fwd_h1, fwd_c1), ...], [(bwd_h1, bwd_c1), ...]``

        This assumes that all the input expression have the same batch size.
        If you batch sentences of the same length together you should pad to
        the longest sequence.

        Args:
            input_sequence (list): Input as a list of
                :py:class:`dynet.Expression` objects
            lengths (list, optional): If the expressions in the sequence are
                batched, but have different lengths, this should contain a list
                of the sequence lengths (default: ``None``)
            left_padded (bool, optional): If the input sequences have different
                lengths they must be padded to the length of longest sequence.
                Use this to specify whether the sequence is left or right
                padded.

        Returns:
            tuple: List of forward and backward recurrent states
                (depends on the recurrent layer)
        """

        # Forward transduction
        forward_states = self.forward_transductor(
            input_sequence,
            lengths=lengths,
            backward=False,
            left_padded=left_padded
        )
        # Backward transduction
        backward_states = self.backward_transductor(
            input_sequence,
            lengths=lengths,
            backward=True,
            left_padded=left_padded
        )
        return forward_states, backward_states
