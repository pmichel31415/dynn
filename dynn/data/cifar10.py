#!/usr/bin/env python3
"""
CIFAR10
^^^^^

Various functions for accessing the
`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ dataset.
"""
import os
import tarfile
import pickle

import numpy as np

from .data_util import download_if_not_there

cifar10_url = "https://www.cs.toronto.edu/~kriz/"
cifar10_file = "cifar-10-python.tar.gz"


def download_cifar10(path=".", force=False):
    """Downloads CIFAR10 from "https://www.cs.toronto.edu/~kriz/cifar.html"

    Args:
        path (str, optional): Local folder (defaults to ".")
        force (bool, optional): Force the redownload even if the files are
            already at ``path``
    """
    # Download the tar.gz file
    download_if_not_there(cifar10_file, cifar10_url, path, force=force)


def _read_file_as_nparray(file_desc):
    d = pickle.load(file_desc, encoding="latin1")
    return d["data"], d["labels"]


def read_cifar10(split, path):
    """Iterates over the CIFAR10 dataset

    Example:

    .. code-block:: python

        for image in read_cifar10("train", "/path/to/cifar10"):
            train(image)

    Args:
        split (str): Either ``"training"`` or ``"test"``
        path (str): Path to the folder containing the ``*-ubyte`` files


    Returns:
        tuple: image, label
    """
    # Adapted from https://gist.github.com/akesling/5358964
    if not (split is "test" or split is "train"):
        raise ValueError("split must be \"train\" or \"test\"")
    abs_path = os.path.join(os.path.abspath(path), cifar10_file)
    with tarfile.open(abs_path) as tar:
        if split is "test":
            filename = "cifar-10-batches-py/test_batch"
            data, labels = _read_file_as_nparray(tar.extractfile(filename))
        else:
            data, labels = [], []
            for i in range(1, 6):
                filename = f"cifar-10-batches-py/data_batch_{i}"
                batch_data, batch_labels = _read_file_as_nparray(
                    tar.extractfile(filename)
                )
                data.append(batch_data)
                labels.append(batch_labels)
            data = np.concatenate(data, axis=0)
            labels = np.concatenate(labels, axis=0)

    images = np.multiply(
        np.asarray(data, dtype=np.uint8).reshape(len(labels), 32, 32, 3),
        1.0 / 255.0
    )

    def get_image(idx): return (images[idx], labels[idx])

    for i in range(len(labels)):
        yield get_image(i)


def load_cifar10(path):
    """Loads the CIFAR10 dataset

    Returns the train and test set, each as a list of images and a list
    of labels. The images are represented as numpy arrays and the labels as
    integers.

    Args:
        path (str): Path to the folder containing the ``*-ubyte.gz`` files

    Returns:
        tuple: train and test sets
    """
    # Read training data
    train = list(read_cifar10("train", path))
    train_img = [img for img, _ in train]
    train_lbl = [lbl for _, lbl in train]
    # Read test data
    test = list(read_cifar10("test", path))
    test_img = [img for img, _ in test]
    test_lbl = [lbl for _, lbl in test]

    return (train_img, train_lbl), (test_img, test_lbl)
