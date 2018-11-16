#!/usr/bin/env python3
import os
import re
from setuptools import setup, find_packages

# Package name
package_name = "dynn"

# Path to this file
here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, "README.md")) as readme:
    README = readme.read()


def read_from_here(*parts):
    with open(os.path.join(here, *parts), 'r') as fp:
        return fp.read()


def find_version():
    """Returns the version from dynn/__init__.py"""
    version_file = read_from_here(package_name, "__init__.py")
    version_match = re.search(
        r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M
    )
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


def read_requirements():
    requirements_text = read_from_here("requirements.txt")
    requirements = [line.strip() for line in requirements_text.split("\n")]
    return requirements


# allow setup.py to be run from any path
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))

setup(
    name=package_name,
    version=find_version(),
    packages=find_packages(exclude=('tests',)),
    license="MIT License",
    description="Neural networks routines for DyNet",
    long_description=read_from_here("README.md"),
    long_description_content_type='text/markdown',
    url="https://github.com/pmichel31415/dynn",
    author="Paul Michel",
    author_email="pmichel1@cs.cmu.edu",
    python_requires=">=3.6",
    install_requires=read_requirements(),
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
    ],
)
