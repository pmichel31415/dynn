#!/usr/bin/env python3
"""
Penn TreeBank
^^^^^^^^^^^^^

Various functions for accessing the
`PTB <http://www.fit.vutbr.cz/~imikolov/rnnlm>`_ dataset used by
`Mikolov et al., 2010 <http://www.fit.vutbr.cz/research/groups/speech/publi/
2010/mikolov_interspeech2010_IS100722.pdf>`_.
"""
import os
import tarfile

from .data_util import download_if_not_there

ptb_url = "http://www.fit.vutbr.cz/~imikolov/rnnlm/"
ptb_file = "simple-examples.tgz"


def download_ptb(path=".", force=False):
    """Downloads the PTB from "http://www.fit.vutbr.cz/~imikolov/rnnlm"

    Args:
        path (str, optional): Local folder (defaults to ".")
        force (bool, optional): Force the redownload even if the files are
            already at ``path``
    """
    download_if_not_there(ptb_file, ptb_url, path, force=force)


def read_ptb(split, path, eos=None):
    """Iterates over the PTB dataset

    Example:

    .. code-block:: python

        for sent in read_ptb("train", "/path/to/ptb"):
            train(sent)

    Args:
        split (str): Either ``"train"``, ``"dev"`` or ``"test"``
        path (str): Path to the folder containing the
            ``simple-examples.tar.gz`` file
        eos (str, optional): Optionally append an end of sentence token to
            each line


    Returns:
        tuple: tree, label
    """
    if not (split is "test" or split is "valid" or split is "train"):
        raise ValueError("split must be \"train\", \"valid\" or \"test\"")
    abs_filename = os.path.join(os.path.abspath(path), ptb_file)

    with tarfile.open(abs_filename) as tar:
        filename = f"./simple-examples/data/ptb.{split}.txt"
        file_obj = tar.extractfile(filename)
        for line in file_obj:
            sent = line.decode("utf-8").strip().split()
            if eos is not None:
                sent.append(eos)
            yield sent


def load_ptb(path, eos=None):
    """Loads the PTB dataset

    Returns the train and test set, each as a list of images and a list
    of labels. The images are represented as numpy arrays and the labels as
    integers.

    Args:
        path (str): Path to the folder containing the
            ``simple-examples.tar.gz`` file
        eos (str, optional): Optionally append an end of sentence token to
            each line


    Returns:
        dict: dictionary mapping the split name to a list of strings
    """
    splits = {}
    for split in ["train", "valid", "test"]:
        splits[split] = list(read_ptb(split, path, eos=eos))

    return splits
