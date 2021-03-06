#!/usr/bin/env python3
"""
Preprocessing functions
^^^^^^^^^^^^^^^^^^^^^^^

Usful functions for preprocessing data
"""
import numpy as np
import sacrebleu
import sacremoses


def lowercase(data):
    """Lowercase text

    Args:
        data (list,str): Data to lowercase (either a string or a list
            [of lists..] of strings)

    Returns:
        list, str: Lowercased data
    """

    if isinstance(data, (list, tuple)):
        return [lowercase(item) for item in data]
    elif isinstance(data, dict):
        return {k: lowercase(v) for k, v in data.items()}
    elif isinstance(data, str):
        return data.lower()
    else:
        raise ValueError("Can only lowercase strings or lists of strings")


def _tokenize(data, tokenizer):
    if isinstance(data, (list, tuple)):
        return [_tokenize(item, tokenizer) for item in data]
    elif isinstance(data, dict):
        return {k: _tokenize(v, tokenizer) for k, v in data.items()}
    elif isinstance(data, str):
        return tokenizer(data)
    else:
        raise ValueError("Can only tokenize strings or lists of strings")


def tokenize(data, tok="space", lang="en"):
    """Tokenize text data.

    There are 5 tokenizers supported:

    - "space": split along whitespaces
    - "char": split in characters
    - "13a": Official WMT tokenization
    - "zh": Chinese tokenization (See ``sacrebleu`` doc)
    - "moses": Moses tokenizer (you can specify lthe language).
       Uses the `sacremoses <https://github.com/alvations/sacremoses>`_

    Args:
        data (list, str): String or list (of lists...) of strings.
        tok (str, optional): Tokenization. Defaults to "space".
        lang (str, optional): Language (only useful for the moses tokenizer).
            Defaults to "en".

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
    elif tok is "moses":
        moses_tok = sacremoses.MosesTokenizer(lang=lang)

        def tokenizer(x): return moses_tok.tokenize(x)
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
    if isinstance(data, (list, tuple)):
        return [normalize(item) for item in data]
    elif isinstance(data, dict):
        return {k: normalize(v) for k, v in data.items()}
    elif isinstance(data, np.ndarray):
        mean = data.mean()
        std = data.std()+1e-20
        return (data-mean)/std
    else:
        raise ValueError(
            "Can only normalize numpy arrays or lists of numpy arrays"
        )
