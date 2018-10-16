#!/usr/bin/env python3
"""
WikiText
^^^^^^^^

Various functions for accessing the
`WikiText <https://einstein.ai/research/blog/the-wikitext-long-term-dependency-
language-modeling-dataset>`_ datasets (WikiText-2 and WikiText-103).
"""
import os
import zipfile

from .data_util import download_if_not_there

wikitext_url = "https://s3.amazonaws.com/research.metamind.io/wikitext/"
wikitext_files = {"2": "wikitext-2-v1.zip", "103": "wikitext-103-v1.zip"}


def download_wikitext(path=".", name="2", force=False):
    """Downloads the WikiText from "http://www.fit.vutbr.cz/~imikolov/rnnlm"

    Args:
        path (str, optional): Local folder (defaults to ".")
        force (bool, optional): Force the redownload even if the files are
            already at ``path``
    """
    if name not in wikitext_files:
        ValueError("Only wikitext `2` and `103` exist")
    filename = wikitext_files[name]
    download_if_not_there(filename, wikitext_url, path, force=force)


def read_wikitext(split, path, name="2", eos=None):
    """Iterates over the WikiText dataset

    Example:

    .. code-block:: python

        for sent in read_wikitext("train", "/path/to/wikitext"):
            train(sent)

    Args:
        split (str): Either ``"train"``, ``"valid"`` or ``"test"``
        path (str): Path to the folder containing the
            ``wikitext-{2|103}-v1.zip`` files
        eos (str, optional): Optionally append an end of sentence token to
            each line


    Returns:
        list: list of words
    """
    if name not in wikitext_files:
        ValueError("Only wikitext `2` and `103` exist")
    if not (split is "test" or split is "valid" or split is "train"):
        raise ValueError("split must be \"train\", \"valid\" or \"test\"")

    abs_filename = os.path.join(os.path.abspath(path), wikitext_files[name])

    with zipfile.ZipFile(abs_filename) as zfile:
        split_file = f"wikitext-{name}/wiki.{split}.tokens"
        with zfile.open(split_file, "r") as file_obj:
            for line in file_obj:
                sent = line.decode("utf-8").strip().split()
                if eos is not None:
                    sent.append(eos)
                yield sent


def load_wikitext(path, eos=None):
    """Loads the WikiText dataset

    Returns the train, validation test set, each as a list of sentences
    (each sentence is a list of words)

    Args:
        path (str): Path to the folder containing the
            ``wikitext-{2|103}-v1.zip`` file
        eos (str, optional): Optionally append an end of sentence token to
            each line


    Returns:
        dict: dictionary mapping the split name to a list of strings
    """
    splits = {}
    for split in ["train", "valid", "test"]:
        splits[split] = list(read_wikitext(split, path, eos=eos))

    return splits
