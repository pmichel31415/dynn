#!/usr/bin/env python3
"""
Data utilities
^^^^^^^^^^^^^^

Helper functions to download and manage datasets.
"""
import os

from urllib.request import urlretrieve
from urllib.parse import urljoin


def download_if_not_there(file, url, path, force=False, local_file=None):
    """Downloads a file from the given url if and only if the file doesn't
    already exist in the provided path or ``force=True``

    Args:
        file (str): File name
        url (str): Url where the file can be found (without the filename)
        path (str): Path to the local folder where the file should be stored
        force (bool, optional): Force the file download (useful if you suspect
            that the file might have changed)
        file (str): File name for the local file (defaults to ``file``)
    """
    # Path to local file
    abs_path = os.path.abspath(path)
    local_file = local_file or file
    local_file_path = os.path.join(abs_path, local_file)
    # Create dir if it doesn't exist
    if not os.path.isdir(abs_path):
        os.mkdir(abs_path)
    # Download if needed
    if force or not os.path.isfile(local_file_path):
        print(f"Downloading file {local_file} to folder {abs_path} from {url}")
        file_url = urljoin(url, file)
        return urlretrieve(file_url, local_file_path)
    return None
