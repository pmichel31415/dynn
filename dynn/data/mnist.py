#!/usr/bin/env python3
"""
MNIST
^^^^^

Various functions for accessing the
`MNIST <http://yann.lecun.com/exdb/mnist/>`_ dataset.
"""
import os
import struct
import gzip
from io import BytesIO
import array

import numpy as np

from .data_util import download_if_not_there

mnist_url = "http://yann.lecun.com/exdb/mnist/"
mnist_files = {
    "train_img": "train-images-idx3-ubyte.gz",
    "train_lbl": "train-labels-idx1-ubyte.gz",
    "test_img": "t10k-images-idx3-ubyte.gz",
    "test_lbl": "t10k-labels-idx1-ubyte.gz",
}


def download_mnist(path=".", force=False):
    """Downloads MNIST from "http://yann.lecun.com/exdb/mnist/"

    Args:
        path (str, optional): Local folder (defaults to ".")
        force (bool, optional): Force the redownload even if the files are
            already at ``path``
    """
    # Download all files sequentially
    for filename in mnist_files.values():
        download_if_not_there(filename, mnist_url, path, force=force)


def read_mnist(split, path):
    """Iterates over the MNIST dataset

    Example:

    .. code-block:: python

        for image in read_mnist("train", "/path/to/mnist"):
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
    abs_path = os.path.abspath(path)
    fname_img = os.path.join(abs_path, mnist_files[f"{split}_img"])
    fname_lbl = os.path.join(abs_path, mnist_files[f"{split}_lbl"])

    with open(fname_lbl, "rb") as zflbl:
        flbl = BytesIO(gzip.decompress(zflbl.read()))
        _, _ = struct.unpack(">II", flbl.read(8))
        data = array.array("B", flbl.read())
        lbl = np.asarray(data, dtype=np.uint8)

    with open(fname_img, "rb") as zfimg:
        fimg = BytesIO(gzip.decompress(zfimg.read()))
        _, _, rows, cols = struct.unpack(">IIII", fimg.read(16))
        data = array.array("B", fimg.read())
        img = np.multiply(
            np.asarray(data, dtype=np.uint8).reshape(len(lbl), rows, cols, 1),
            1.0 / 255.0
        )

    def get_img(idx): return (img[idx], lbl[idx])

    for i in range(len(lbl)):
        yield get_img(i)


def load_mnist(path):
    """Loads the MNIST dataset

    Returns MNIST as a dictionary.

    Example:

    .. code-block:: python

        mnist = load_mnist(".")
        # Train images and labels
        train_imgs, train_labels = mnist["train"]
        # Test images and labels
        test_imgs, test_labels = mnist["test"]

    The images are represented as numpy arrays and the labels as
    integers.

    Args:
        path (str): Path to the folder containing the ``*-ubyte.gz`` files

    Returns:
        dict: MNIST dataset
    """
    splits = {}
    # Read data
    for split in ["train", "test"]:
        data = list(read_mnist("train", path))
        images = [img for img, _ in data]
        labels = [lbl for _, lbl in data]
        splits[split] = (images, labels)

    return splits
