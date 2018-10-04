#!/usr/bin/env python3
"""
Preprocessing functions
^^^^^^^^^^^^^^^^^^^^^^^

Usful functions for preprocessing data
"""
import numpy as np


def lowercase(data):
    """Lowercase text

    Args:
        data (list,str): Data to lowercase (either a string or a list
            [of lists..] of strings)

    Returns:
        list, str: Lowercased data
    """

    if isinstance(data, list):
        return [lowercase(item) for item in data]
    elif isinstance(data, str):
        return data.lower()
    else:
        raise ValueError("Can only lowercase strings or lists of strings")


def normalize(data):
    """Normalize the data to mean 0 std 1

    Args:
        data (list,np.ndarray): data to normalize

    Returns:
        list,np.array: Normalized data
    """

    if isinstance(data, list):
        return [normalize(item) for item in data]
    elif isinstance(data, np.ndarray):
        mean = data.mean()
        std = data.std()+1e-20
        return (data-mean)/std
    else:
        raise ValueError(
            "Can only normalize numpy arrays or lists of numpy arrays"
        )
