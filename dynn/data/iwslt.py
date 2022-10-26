#!/usr/bin/env python3
"""
IWSLT
^^^^^

Various functions for accessing the IWSLT translation datasets
"""
import os
import tarfile
import re

from .data_util import download_if_not_there
from .dictionary import EOS_TOKEN

iwslt_url = "https://wit3.fbk.eu/archive"
supported = {
    "2016.de-en": {
        "train": "train.tags.de-en",
        "dev": "IWSLT16.TED.tst2013.de-en",
        "test": "IWSLT16.TED.tst2014.de-en"
    },
    "2016.fr-en": {
        "train": "train.tags.fr-en",
        "dev": "IWSLT16.TED.tst2013.fr-en",
        "test": "IWSLT16.TED.tst2014.fr-en"
    },
}
supported_string = ", ".join(supported)


# Regex for filtering metadata
is_meta = re.compile(r"^<.*")
not_seg = re.compile(r"^(?!<seg.).*")
eval_segment = re.compile(r"^<seg[^>]*>(.*)</seg>")


def local_dir(year, langpair):
    return f"iwslt{year}.{langpair}"


def download_iwslt(path=".", year="2016", langpair="de-en", force=False):
    """Downloads the IWSLT from "https://wit3.fbk.eu/archive/"

    Args:
        path (str, optional): Local folder (defaults to ".")
        year (str, optional): IWSLT year (for now only 2016 is supported)
        langpair (str, optional): ``src-tgt`` language pair (for now only
            ``{de,fr}-en`` are supported)
        force (bool, optional): Force the redownload even if the files are
            already at ``path``
    """
    src, tgt = langpair.split("-")
    langpair_url = f"{iwslt_url}/{year}-01//texts/{src}/{tgt}/"
    langpair_file = f"{langpair}.tgz"
    local_file = f"iwslt{year}.{langpair}.tgz"
    downloaded = download_if_not_there(
        langpair_file,
        langpair_url,
        path,
        force=force,
        local_file=local_file
    )
    # Extract
    if downloaded:
        abs_filename = os.path.join(os.path.abspath(path), local_file)
        # Create target dir
        directory = local_dir(year, langpair)
        root_path = os.path.join(os.path.abspath(path), directory)
        if not os.path.isdir(root_path):
            os.mkdir(root_path)
        # Extract

        def members(tf):
            L = len(f"{langpair}/")
            for member in tf.getmembers():
                if member.path.startswith(f"{langpair}/"):
                    member.path = member.path[L:]
                    yield member
        with tarfile.open(abs_filename) as tar:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, root_path, members=members(tar))


def read_iwslt(
    split,
    path,
    year="2016",
    langpair="de-en",
    src_eos=None,
    tgt_eos=EOS_TOKEN
):
    """Iterates over the IWSLT dataset

    Example:

    .. code-block:: python

        for src, tgt in read_iwslt("train", "/path/to/iwslt"):
            train(src, tgt)

    Args:
        split (str): Either ``"train"``, ``"dev"`` or ``"test"``
        path (str): Path to the folder containing the ``.tgz`` file
        year (str, optional): IWSLT year (for now only 2016 is supported)
        langpair (str, optional): ``src-tgt`` language pair (for now only
            ``{de,fr}-en`` are supported)
        src_eos (str, optional): Optionally append an end of sentence token to
            each source line.
        tgt_eos (str, optional): Optionally append an end of sentence token to
            each target line.

    Returns:
        tuple: Source sentence, Target sentence
    """
    if not (split is "test" or split is "dev" or split is "train"):
        raise ValueError("split must be \"train\", \"dev\" or \"test\"")
    is_train = split is "train"
    # Languages
    src, tgt = langpair.split("-")
    # Local dir
    directory = local_dir(year, langpair)
    root_path = os.path.join(os.path.abspath(path), directory)
    # Retrieve source/target file
    prefix = supported[f"{year}.{langpair}"][split]
    src_file = os.path.join(root_path, f"{prefix}.{src}")
    tgt_file = os.path.join(root_path, f"{prefix}.{tgt}")
    if not is_train:
        src_file = f"{src_file}.xml"
        tgt_file = f"{tgt_file}.xml"
    src_obj = open(src_file, encoding="utf-8")
    tgt_obj = open(tgt_file, encoding="utf-8")
    # Read lines
    for src_l, tgt_l in zip(src_obj, tgt_obj):
        # src_l, tgt_l = src_bytes.decode(), tgt_bytes.decode()
        # Skip metadata
        if is_train and is_meta.match(src_l) and is_meta.match(tgt_l):
            continue
        if not is_train:
            src_seg = eval_segment.match(src_l)
            tgt_seg = eval_segment.match(tgt_l)
            if src_seg is None or tgt_seg is None:
                continue
            # Strip segments
            src_l = src_seg.group(1)
            tgt_l = tgt_seg.group(1)
        # Split
        src_l = src_l.strip()
        tgt_l = tgt_l.strip()
        # Append eos maybe
        if src_eos is not None:
            src_l += src_eos
        if tgt_eos is not None:
            tgt_l += tgt_eos
        # Return
        yield src_l, tgt_l

    src_obj.close()
    tgt_obj.close()


def load_iwslt(
    path,
    year="2016",
    langpair="de-en",
    src_eos=None,
    tgt_eos=EOS_TOKEN
):
    """Loads the IWSLT dataset

    Returns the train, dev and test set, each as lists of source and target
    sentences.

    Args:
        path (str): Path to the folder containing the ``.tgz`` file
        year (str, optional): IWSLT year (for now only 2016 is supported)
        langpair (str, optional): ``src-tgt`` language pair (for now only
            ``{de,fr}-en`` are supported)
        src_eos (str, optional): Optionally append an end of sentence token to
            each source line.
        tgt_eos (str, optional): Optionally append an end of sentence token to
            each target line.


    Returns:
        tuple: train, dev and test sets
    """
    if f"{year}.{langpair}" not in supported:
        raise ValueError(
            f"{year}.{langpair} not supported. "
            f"Supported datasets are {supported_string}"
        )
    splits = []
    for split in ["train", "dev", "test"]:
        data = list(read_iwslt(
            split,
            path,
            year=year,
            langpair=langpair,
            src_eos=src_eos,
            tgt_eos=tgt_eos
        ))
        src_data = [src for src, _ in data]
        tgt_data = [tgt for _, tgt in data]
        splits.append((src_data, tgt_data))

    return tuple(splits)
