#!/usr/bin/env python3
"""
Preprocessing functions
^^^^^^^^^^^^^^^^^^^^^^^

Usful functions for preprocessing data
"""
import numpy as np
import sacrebleu


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


def _tokenize(data, tokenizer):
    if isinstance(data, list):
        return [_tokenize(item) for item in data]
    elif isinstance(data, str):
        return tokenizer(data)
    else:
        raise ValueError("Can only tokenize strings or lists of strings")


def tokenize(data, tok="space"):
    """Tokenize text data.

    There are 5 tokenizers supported:

    - "space": split along whitespaces
    - "char": split in characters
    - "13a": Official WMT tokenization
    - "zh": Chinese tokenization (See ``sacrebleu`` doc)

    Args:
        data (list, str): String or list (of lists...) of strings
        tok (str, optional): Tokenization. Defaults to "space".

    Returns:
        list, str: Tokenized data
    """

    if tok is "space":
        def tokenizer(x): return x.split()
    elif tok is "char":
        def tokenizer(x): return list(x)
    elif tok is "13a":
        def tokenizer(x): return sacrebleu.tokenize_13a(x).split(" ")
    elif tok is "zh":
        def tokenizer(x): return sacrebleu.tokenize_zh(x).split(" ")
    else:
        raise ValueError(f"Unknown tokenizer {tok}")
    return _tokenize(data, tokenizer)


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
