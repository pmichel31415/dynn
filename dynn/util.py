#!/usr/bin/env python3
"""
Utility functions
=================
"""

from collections import Iterable

import numpy as np
import dynet as dy

from . import operations


def _default_value(argument, default):
    """Returns ``default`` if ``argument`` is ``None``"""
    if argument is None:
        return default
    else:
        return argument


def list_to_matrix(l):
    """Transforms a list of N vectors of dimension d into a (N, d) matrix"""
    if isinstance(l, list):
        return operations.stack(l, d=0)
    else:
        return l


def matrix_to_image(M):
    """Transforms a matrix (d1, d2) into an 'image' with one channel
    (d1, d2, 1)"""
    dim, bsize = M.dim()
    if len(dim) == 1:
        return dy.reshape(M, (dim[0], 1, 1), batch_size=bsize)
    elif len(dim) == 2:
        d1, d2 = dim
        return dy.reshape(M, (d1, d2, 1), batch_size=bsize)
    elif len(dim) == 3:
        return M
    else:
        raise ValueError('Cannot convert tensor of order >3 to image')


def image_to_matrix(M):
    """Transforms an 'image' with one channel (d1, d2, 1) into a matrix
    (d1, d2)"""
    dim, bsize = M.dim()
    if len(dim) == 3:
        d1, d2, d3 = dim
        assert d3 == 1, 'Input image has more than 1 channel'
        return dy.reshape(M, (d1, d2), batch_size=bsize)
    else:
        return M


def conditional_dropout(x, dropout_rate, flag):
    """This helper function applies dropout only if the flag
    is set to ``True`` and the ``dropout_rate`` is positive.

    Args:
        x (:py:class:`dynet.Expression`): Input expression
        dropout_rate (float): Dropout rate
        flag (bool): Setting this to false ensures that dropout is never
            applied (for testing for example)
    """
    if dropout_rate > 0 and flag:
        return dy.dropout(x, dropout_rate)
    else:
        return x


def sin_embeddings(length, dim, transposed=False):
    """Returns sinusoidal position encodings.

    As described in `Vaswani et al. (2017) <https://arxiv.org/abs/1706.03762>`_

    Specifically this return a ``length x dim`` matrix :math:`PE` such that
    :math:`PE[p, 2i]=\sin(\\frac{p}/{1000^{\\frac{2i}{dim}}})` and
    :math:`PE[p, 2i+1]=\cos(\\frac{p}/{1000^{\\frac{2i}{dim}}})`

    Args:
        length (int): Length
        dim (int): Dimension of the embeddings
    """
    # Scale for each dimension
    dim_scale = 2 * np.floor_divide(np.arange(dim), 2) / dim
    dim_scale = np.float_power(10000.0, dim_scale).reshape(1, -1)
    # Phase to change sine to cosine every other dim
    phase = np.zeros((1, dim))
    phase[0, 1::2] = np.pi / 2
    # Position value
    pos = np.arange(length).reshape(-1, 1)
    # Embeddings
    embeds = np.sin(pos / dim_scale + phase)
    # Return
    if transposed:
        return embeds.T
    else:
        return embeds

# Masking


def apply_mask(x, m, val):
    return dy.cmult(x, m) + val


def _mask_batch(x, mask, value):
    if isinstance(x, dy.Expression):
        # At the bottom of the recursion x is an expression
        # Reshape the mask
        mask_dim = tuple([1] * len(x.dim()[0]))
        batch_size = mask.dim()[1]
        reshaped_mask = dy.reshape(mask, mask_dim, batch_size=batch_size)
        reshaped_value = dy.reshape(value, mask_dim, batch_size=batch_size)
        # Apply the mask
        return apply_mask(x, reshaped_mask, reshaped_value)
    else:
        # Otherwise iterate
        output = []
        for expression in x:
            output.append(_mask_batch(expression, mask, value))
        return output


def mask_batches(x, mask, value=0.0):
    """Apply a mask to the batch dimension

    Args:
        x (list, :py:class:`dynet.Expression`): The expression we want to mask.
            Either a :py:class:`dynet.Expression` or a list thereof with the
            same batch dimension.
        mask (np.array, list, :py:class:`dynet.Expression`): The mask. Either
            a list, 1d numpy array or :py:class:`dynet.Expression`.
        value (float): Mask value
    """
    # Check x's type
    if not isinstance(x, (dy.Expression, Iterable)):
        raise ValueError("x must be a dynet.Expression or an Iterable")
    # Get the mask expression
    if not isinstance(mask, dy.Expression):
        mask_vals = mask[:].astype(float)
        mask_vals[mask == 1] = value
        mask = dy.inputTensor(1-mask, batched=True)
        mask_vals = dy.inputTensor(mask_vals, batched=True)
    # Check that the mask has valid dimensions
    if any(dimension != 1 for dimension in mask.dim()[0]):
        raise ValueError(
            f"Batch masks should have all dimensions == 1 except for "
            f"the batch dimension, got {mask.dim()} instead."
        )
    # Actually do the masking
    return _mask_batch(x, mask, mask_vals)


def _generate_mask(
    step,
    max_length,
    batch_size,
    lengths,
    left_padded
):
    lengths = np.asarray(lengths)
    # This helper function computes masks
    step_number = np.full(batch_size, step)
    # Compute a mask with value 1 if the sequence is not over
    # and 0 if it is. The behaviour is different depending on
    # left/right padding
    if left_padded:
        # If the sequence is left padded
        outside_sequence = (step_number >= lengths)
    else:
        # If the sequence is right padded
        outside_sequence = step_number < (max_length - lengths)
    return outside_sequence.astype(int)


def _should_mask(step, min_length, max_length, left_padded):
    # Decide whether there is any need to do masking depending
    # on the current position
    # `step` is the next input position
    if left_padded:
        return step >= min_length
    else:
        return step + min_length < max_length


def num_params(pc, params=True, lookup_params=True):
    """Number of parameters in a given ParameterCollection"""
    N = 0
    # All parameters
    if params:
        for p in pc.parameters_list():
            p_size = 1
            for d in p.shape():
                p_size *= d
            N += p_size
    # All lookup parameters
    if lookup_params:
        for lp in pc.lookup_parameters_list():
            lp_size = 1
            for d in lp.shape():
                lp_size *= d
            N += lp_size
    return N
