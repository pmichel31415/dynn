#!/usr/bin/env python3
import os
from setuptools import setup, find_packages

with open(os.path.join(os.path.dirname(__file__), "README.md")) as readme:
    README = readme.read()

# allow setup.py to be run from any path
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))

setup(
    name="dynn",
    version="0.1",
    packages=find_packages(exclude=('tests',)),
    license="MIT License",
    description="Neural networks routines for DyNet",
    long_description=README,
    url="https://github.com/pmichel31415/dynn",
    author="Paul Michel",
    author_email="pmichel1@cs.cmu.edu",
    python_requires=">=3.6",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
    ],
)
