#!/usr/bin/env python3
"""
Amazon elec dataset
^^^^^^^^^^^^^^^^^^^

Various functions for accessing the
`Amazon Reviews <http://riejohnson.com/cnn_data.html>`_ dataset.
"""
import os
import tarfile

from .data_util import download_if_not_there

amazon_url = "http://riejohnson.com/software/"
amazon_file = "elec2.tar.gz"


def download_amazon(path=".", force=False):
    """Downloads the Amazon from "http://riejohnson.com/software/"

    Args:
        path (str, optional): Local folder (defaults to ".")
        force (bool, optional): Force the redownload even if the files are
            already at ``path``
    """
    download_if_not_there(amazon_file, amazon_url, path, force=force)


def _valid_size(size):
    valid_sizes = {"200k", "100k", "50k", "25k", "10k", "05k", "02k"}
    if size not in valid_sizes:
        raise ValueError(
            f"Size must be in {', '.join(valid_sizes)} "
            f"(got {size})"
        )


def read_amazon(split, path, tok=True, size="200k"):
    """Iterates over the Amazon dataset

    Example:

    .. code-block:: python

        for review, label in read_amazon("train", "/path/to/amazon"):
            train(review, label)

    Args:
        split (str): Either ``"train"``, ``"dev"`` or ``"test"``
        path (str): Path to the folder containing the
            ``elec2.tar.gz`` files


    Returns:
        tuple: review, label
    """
    if split is "train":
        _valid_size(size)
        labels_filename = f"elec/elec-{size}-{split}.cat"
        reviews_filename = f"elec/elec-{size}-{split}.txt"
    elif split is "test":
        labels_filename = f"elec/elec-{split}.cat"
        reviews_filename = f"elec/elec-{split}.txt"
    else:
        raise ValueError("split must be \"train\" or \"test\"")

    if tok:
        reviews_filename = f"{reviews_filename}.tok"

    abs_filename = os.path.join(os.path.abspath(path), amazon_file)

    reviews = []
    labels = []

    with tarfile.open(abs_filename) as tar:
        with tar.extractfile(reviews_filename) as f:
            for line in f:
                review = line.decode().strip().split()
                reviews.append(review)

        with tar.extractfile(labels_filename) as f:
            for line in f:
                label = int(line.strip()) - 1
                labels.append(label)

    def get_sample(idx): return reviews[idx], labels[i]

    for i in range(len(labels)):
        yield get_sample(i)


def load_amazon(path, tok=True, size="200k"):
    """Loads the Amazon dataset

    Returns the train, dev and test sets in a dictionary, each as a tuple of
    containing the reviews and the labels.

    Args:
        path (str): Path to the folder containing the
            ``elec2.tar.gz`` file

    Returns:
        dict: Dictionary containing the train and test sets
            (dictionary of review/labels tuples)
    """
    splits = {}
    for split in ["train", "test"]:
        data = list(
            read_amazon(split, path, tok=tok, size=size)
        )
        reviews = [review for review, _ in data]
        labels = [lbl for _, lbl in data]
        splits[split] = (reviews, labels)

    return splits
