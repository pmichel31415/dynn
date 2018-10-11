#!/usr/bin/env python3
"""
Sequence transduction layers
============================

Sequence transduction layers take in a sequence of expression and runs one
layer over each input. They can be feed-forward (each input is treated
independently, eg. :py:class:`Transduction`) or recurrent
(the output at one step depends on the output at the previous step, eg.
:py:class:`Unidirectional`).
"""

from ..util import _generate_mask, _should_mask, mask_batches, _default_value
from .base_layers import BaseLayer


class Transduction(BaseLayer):
    """Feed forward transduction layer

    This layer runs one cell on a sequence of inputs and returns the list of
    outputs. Calling it is equivalent to calling:

    .. code-block:: python

        [layer(x) for x in input_sequence]

    Args:
        cell (:py:class:`base_layers.BaseLayer`): The
            recurrent cell to use for transduction
    """

    def __init__(self, layer):

        self.layer = layer

    def init(self, *args, **kwargs):
        """Passes its arguments to the recurrent layer"""
        self.layer.init(*args, **kwargs)

    def __call__(
        self,
        input_sequence,
    ):
        """Runs the layer over the input

        The output is a list of the output of the layer at each step

        Args:
            input_sequence (list): Input as a list of
                :py:class:`dynet.Expression` objects

        Returns:
            list: List of recurrent states (depends on the recurrent layer)
        """
        output_sequence = [self.layer(x) for x in input_sequence]
        return output_sequence


class SequenceMaskingLayer(BaseLayer):
    """Masks a sequence of batched expressions according to each batch
    element's length

    This layer applies a mask value to the elements of a sequence of batched
    expressions which correspond to padding tokens. Typically if you batch a
    sequence of size 2 and a sequence of size 3 you will pad the first sequence
    to obtain a list of 3 expresions of batch size 2. This layer will mask the
    batch element of the last expression corresponding to the padding token in
    the 1st sequence.

    This is useful when doing attention or max-pooling on padded sequences when
    you want to mask padding tokens with :math:`-\infty` to ensure that they
    are ignored.


    Args:
        mask_value (float, optional): The value to use for masking
        left_padded (bool, optional): If the input sequences have different
            lengths they must be padded to the length of longest sequence.
            Use this to specify whether the sequence is left or right
            padded.
    """

    def __init__(self, mask_value=0.0, left_padded=True):
        self.mask_value = mask_value
        self.left_padded = left_padded

    def __call__(
            self,
            input_sequence,
            lengths,
            left_padded=None):
        """Runs the layer over the input

        The output is a list of the output of the layer at each step

        Args:
            input_sequence (list): Input as a list of
                :py:class:`dynet.Expression` objects
            lengths (list, optional): If the expressions in the sequence are
                batched, but have different lengths, this should contain a list
                of the sequence lengths (default: ``None``)
            left_padded (bool, optional): If the input sequences have different
                lengths they must be padded to the length of longest sequence.
                Use this to specify whether the sequence is left or right
                padded. Overwrites the value given in the constructor.

        Returns:
            list: List of masked expression
        """
        # Max length/ batch size
        max_length = len(input_sequence)
        batch_size = input_sequence[0].dim()[1]
        # Overwrite left padding flag
        left_padded = self.left_padded if left_padded is None else left_padded
        # Start transducing
        output_sequence = []
        for step, x in enumerate(input_sequence):
            # Perform masking if we need to
            if _should_mask(step, min(lengths), max_length, left_padded):
                # Generate the max depending on the position, padding, etc...
                mask = _generate_mask(
                    step, max_length, batch_size, lengths, left_padded
                )
                # Apply it
                output_sequence.append(mask_batches(
                    x, mask, value=self.mask_value))
            else:
                output_sequence.append(x)

        return output_sequence
        output_sequence = [self.layer(x) for x in input_sequence]
        return output_sequence


class Unidirectional(BaseLayer):
    """Unidirectional transduction layer

    This layer runs a recurrent cell on a sequence of inputs and produces
    resulting the sequence of recurrent states.

    Example:

    .. code-block:: python

        # LSTM cell
        lstm_cell = dynn.layers.LSTM(dy.ParameterCollection(), 10, 10)
        # Transduction layer
        lstm = dynn.layers.Unidirectional(lstm_cell)
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
        output_only (bool, optional): Only return the sequence of outputs
            instead of the sequence of states.
    """

    def __init__(self, cell, output_only=False):
        super(Unidirectional, self).__init__("unidirectional")
        self.cell = cell
        self.output_only = output_only

    def init(self, *args, **kwargs):
        """Passes its arguments to the recurrent layer"""
        self.cell.init(*args, **kwargs)

    def __call__(
        self,
        input_sequence,
        backward=False,
        lengths=None,
        left_padded=True,
        output_only=None,
        initial_state=None,
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
            output_only (bool, optional): Only return the sequence of outputs
                instead of the sequence of states. Overwrites the value given
                in the constructor.
            initial_state (:py:class:`dy.Expression`, optional): Overridies the
                default initial state of the recurrent cell

        Returns:
            list: List of recurrent states (depends on the recurrent layer)
        """
        # Dimensions of the input sequence
        batch_size = input_sequence[0].dim()[1]
        max_length = len(input_sequence)
        min_length = max_length if lengths is None else min(lengths)
        # Reverse the sequence for backward transduction
        if backward:
            input_sequence = reversed(input_sequence)
            # Padding is also reversed
            left_padded = not left_padded
        # Initial recurrent state provided by the recurrent cell
        if initial_state is None:
            state = self.cell.initial_value(batch_size=batch_size)
        else:
            state = initial_state
        # Start transducing
        output_sequence = []
        for step, x in enumerate(input_sequence):
            # Call the recurrent cell
            # The arguments to the recurrent layer will always be
            # recurrent state first and then input
            state = self.cell(x, *state)
            # Perform masking if we need to
            # TODO: masking -> interpolation with initial state
            if _should_mask(step, min_length, max_length, left_padded):
                # Generate the max depending on the position, padding, etc...
                mask = _generate_mask(
                    step, max_length, batch_size, lengths, left_padded
                )
                # Apply it
                state = mask_batches(state, mask, value=0)
            # Add the masked state to the output
            output_sequence.append(state)
        # Reverse the sequence for backward transduction
        if backward:
            output_sequence = output_sequence[::-1]
        # Only return outputs
        if _default_value(output_only, self.output_only):
            output_sequence = [self.cell.get_output(state)
                               for state in output_sequence]
        return output_sequence


class Bidirectional(BaseLayer):
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
        bilstm = dynn.layers.Bidirectional(fwd_lstm_cell, bwd_lstm_cell)
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
        output_only (bool, optional): Only return the sequence of outputs
            instead of the sequence of states.
    """

    def __init__(self, forward_cell, backward_cell, output_only=False):
        super(Bidirectional, self).__init__("bidirectional")
        self.forward_transductor = Unidirectional(
            forward_cell, output_only)
        self.backward_transductor = Unidirectional(
            backward_cell, output_only)
        self.output_only = output_only

    def init(self, *args, **kwargs):
        """Passes its arguments to the recurrent layers"""
        self.forward_transductor.init(*args, **kwargs)
        self.backward_transductor.init(*args, **kwargs)

    def __call__(
        self,
        input_sequence,
        lengths=None,
        left_padded=True,
        output_only=None,
        fwd_initial_state=None,
        bwd_initial_state=None,
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
            output_only (bool, optional): Only return the sequence of outputs
                instead of the sequence of states. Overwrites the value given
                in the constructor.
            fwd_initial_state (:py:class:`dy.Expression`, optional): Overridies
                the default initial state of the forward recurrent cell.
            bwd_initial_state (:py:class:`dy.Expression`, optional): Overridies
                the default initial state of the backward recurrent cell.

        Returns:
            tuple: List of forward and backward recurrent states
                (depends on the recurrent layer)
        """

        # Forward transduction
        forward_states = self.forward_transductor(
            input_sequence,
            lengths=lengths,
            backward=False,
            left_padded=left_padded,
            output_only=output_only,
            initial_state=fwd_initial_state,
        )
        # Backward transduction
        backward_states = self.backward_transductor(
            input_sequence,
            lengths=lengths,
            backward=True,
            left_padded=left_padded,
            output_only=output_only,
            initial_state=bwd_initial_state,
        )
        return forward_states, backward_states
