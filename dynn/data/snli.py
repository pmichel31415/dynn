#!/usr/bin/env python3
"""
Stanford Natural Language Inference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Various functions for accessing the
`SNLI <https://nlp.stanford.edu/projects/snli/>`_ dataset.
"""
import os
import zipfile

from .data_util import download_if_not_there
from .trees import Tree

snli_url = "https://nlp.stanford.edu/projects/snli/"
snli_file = "snli_1.0.zip"


def download_snli(path=".", force=False):
    """Downloads the SNLI from "https://nlp.stanford.edu/projects/snli/"

    Args:
        path (str, optional): Local folder (defaults to ".")
        force (bool, optional): Force the redownload even if the files are
            already at ``path``
    """
    download_if_not_there(snli_file, snli_url, path, force=force)


def read_snli(split, path, terminals_only=True, binary=False):
    """Iterates over the SNLI dataset

    Example:

    .. code-block:: python

        for tree, label in read_snli("train", "/path/to/snli"):
            train(tree, label)

    Args:
        split (str): Either ``"train"``, ``"dev"`` or ``"test"``
        path (str): Path to the folder containing the
            ``snli_1.0.zip`` files
        terminals_only (bool): Only return the terminals and not the trees


    Returns:
        tuple: tree, label
    """
    if not (split is "test" or split is "dev" or split is "train"):
        raise ValueError("split must be \"train\" or \"test\"")
    abs_filename = os.path.join(os.path.abspath(path), snli_file)

    labels = []
    premises = []
    hypotheses = []

    with zipfile.ZipFile(abs_filename) as zfile:
        with zfile.open(f"snli_1.0/snli_1.0_{split}.txt", "r") as f:
            # Skip headers
            f.readline()
            for line in f:
                fields = line.decode().strip().split("\t")
                label, premise, hypothesis = fields[:3]
                # Skip unknown labels
                if label is "-":
                    continue
                # Get the binary parses and labels
                premise_tree = Tree.from_string(premise, labelled=False)
                hypothesis_tree = Tree.from_string(hypothesis, labelled=False)
                # Whether we want trees or only terminals
                if terminals_only:
                    premise_tree = premise_tree.leaves()
                    hypothesis_tree = hypothesis_tree.leaves()

                premises.append(premise_tree)
                hypotheses.append(hypothesis_tree)
                labels.append(label)

    def get_sample(idx): return premises[i], hypotheses[i], labels[idx]

    for i in range(len(labels)):
        yield get_sample(i)


def load_snli(path, terminals_only=True, binary=False):
    """Loads the SNLI dataset

    Returns the train, dev and test sets in a dictionary, each as a tuple of
    containing the trees and the labels.

    Args:
        path (str): Path to the folder containing the
            ``snli_1.0.zip`` file
        terminals_only (bool): Only return the terminals and not the trees

    Returns:
        dict: Dictionary containing the train, dev and test sets
            (tuple of tree/labels tuples)
    """
    splits = {}
    for split in ["train", "dev", "test"]:
        data = list(
            read_snli(split, path, terminals_only=terminals_only)
        )
        premises = [premise for premise, _, _ in data]
        hypotheses = [hypothesis for _, hypothesis, _ in data]
        labels = [lbl for _, _, lbl in data]
        splits[split] = (premises, hypotheses, labels)

    return splits
