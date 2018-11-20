#!/usr/bin/env python3
"""
Command line utilities
======================
"""


def add_dynet_args(parser, new_group=True):
    """Adds dynet command line arguments to an
    :py:class:`argparse.ArgumentParser`

    You can apply this to your argument parser so that it doesn't throw an
    error when you add command line arguments for dynet. For a description
    of the arguments available for dynet, see `the official documentation
    <https://dynet.readthedocs.io/en/latest/commandline.html>`_

    Args:
        parser (:py:class:`argparse.ArgumentParser`): Your argument parser.
        new_group (bool, optional): Add the arguments in a specific argument
            group (default: `True`)
    """
    if new_group:
        parser = parser.add_argument_group("DyNet specific arguments")
    parser.add_argument("--dynet-gpu", action="store_true")
    parser.add_argument("--dynet-gpus", type=int)
    parser.add_argument("--dynet-devices", type=str)
    parser.add_argument("--dynet-mem", type=str)
    parser.add_argument("--dynet-autobatch", type=int)
    parser.add_argument("--dynet-weight-decay", type=float)
    parser.add_argument("--dynet-profiling", type=int)
