#!/usr/bin/env python3
"""
Functions for Dataset Caching
=============================
"""
import os
import pickle


def cached_to_file(filename):
    """Decorator to cache the output of a function to a file

    Sometimes your workflow will contain functions that are executed once but
    take a lot of time (typically data preprocessing). This can be annoying
    when eg. running multiple experiments different parameters. This decorator
    provides a solution by running the function once, then saving its output
    to a file. Next time you called this function, and unless the file in
    question has been deleted, the function will just read its result from the
    file instead of recomputing everything.

    Caveats:
    - By default if you call the decorated function with different arguments,
      this will still load the cached output from the first function call with
      the *original arguments*. You need to add the `update_cache=True`
      keyword argument to force the function to be rerun. Incidentally the
      decorated function should not have an argument named `update_cache`.
    - The serialization is done with pickle, so:
        1. it isn't super secure (if you care about these things)
        2. it only handles functions where the outputs can be pickled
           (for now). Typically this wouldn't work for dynet objects.

    Example usage:

    .. code-block:: python

        @cached_to_file("preprocessed_data.bin")
        def preprocess(raw_data):
            # do a lot of preprocessing

        # [...] do something else

        # This first call will run the function and pickle its output to
        # "preprocessed_data.bin" (and return the output)
        data = preprocess(raw_data)

        # [...] do something else, or maybe rerun the program

        # This will just load the output from "preprocessed_data.bin"
        data = preprocess(raw_data)

        # [...] do something else, or maybe rerun the program

        # This will force the function to be rerun and the cached output to be
        # updated. You should to that if for example the arguments of
        # `preprocess` are expected to change
        data = preprocess(raw_data, update_cache=True)

    Args:
        filename (str): Name of the file where the cached output should
            be saved to.
    """

    def _load_cached_output(func):
        def wrapped_func(*args, update_cache=False, **kwargs):
            if not os.path.isfile(filename) or update_cache:
                # If the cached output doesn't exist, do all the processing
                output = func(*args, **kwargs)
                with open(filename, "wb") as f:
                    pickle.dump(output, f)
            else:
                # Other unpickle the preprocessed output
                print(f"Loading cached output of {func.__name__} from "
                      f"{filename}")
                with open(filename, "rb") as f:
                    output = pickle.load(f)
            return output
        return wrapped_func

    return _load_cached_output
