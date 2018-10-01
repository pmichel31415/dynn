#!/usr/bin/env python3
"""
Stanford Sentiment TreeBank
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Various functions for accessing the
`SST <https://nlp.stanford.edu/sentiment/index.html>`_ dataset.
"""
import os
import zipfile

from .data_util import download_if_not_there
from .trees import Tree

sst_url = "https://nlp.stanford.edu/sentiment/"
sst_file = "trainDevTestTrees_PTB.zip"


def download_sst(path=".", force=False):
    """Downloads the SST from "https://nlp.stanford.edu/sentiment/"

    Args:
        path (str, optional): Local folder (defaults to ".")
        force (bool, optional): Force the redownload even if the files are
            already at ``path``
    """
    download_if_not_there(sst_file, sst_url, path, force=force)


def read_sst(split, path, terminals_only=True, binary=False):
    """Iterates over the SST dataset

    Example:

    .. code-block:: python

        for tree, label in read_sst("train", "/path/to/sst"):
            train(tree, label)

    Args:
        split (str): Either ``"train"``, ``"dev"`` or ``"test"``
        path (str): Path to the folder containing the
            ``trainDevTestTrees_PTB.zip`` files
        terminals_only (bool): Only return the terminals and not the tree
        binary (bool): Binary SST (only positive and negative labels).
            Neutral lables are discarded


    Returns:
        tuple: tree, label
    """
    if not (split is "test" or split is "dev" or split is "train"):
        raise ValueError("split must be \"train\" or \"test\"")
    abs_filename = os.path.join(os.path.abspath(path), sst_file)

    trees = []
    labels = []

    with zipfile.ZipFile(abs_filename) as zfile:
        with zfile.open(f"trees/{split}.txt", "r") as f:
            for line in f:
                tree = Tree.from_string(line.decode())
                # Treat the case of binary SST
                # TODO: handle subtree labels
                if binary:
                    if tree.label == 2:
                        continue
                    else:
                        tree.label = int(tree.label / 2.5)
                if terminals_only:
                    trees.append(tree.leaves())
                else:
                    trees.append(tree)

                labels.append(tree.label)

    def get_sample(idx): return trees[idx], labels[i]

    for i in range(len(labels)):
        yield get_sample(i)


def load_sst(path, terminals_only=True, binary=False):
    """Loads the SST dataset

    Returns the train and test set, each as a list of images and a list
    of labels. The images are represented as numpy arrays and the labels as
    integers.

    Args:
        path (str): Path to the folder containing the
            ``trainDevTestTrees_PTB.zip`` file
        terminals_only (bool): Only return the terminals and not the tree
        binary (bool): Binary SST (only positive and negative labels).
            Neutral lables are discarded

    Returns:
        tuple: train, dev and test sets (tuple of tree/labels tuples)
    """
    splits = []
    # TODO: binary labels
    for split in ["train", "dev", "test"]:
        data = list(
            read_sst(split, path, terminals_only=terminals_only, binary=binary)
        )
        trees = [tree for tree, _ in data]
        labels = [lbl for _, lbl in data]
        splits.append([trees, labels])

    return tuple(splits)
