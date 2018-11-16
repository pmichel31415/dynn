#!/usr/bin/env python3
"""
Loading/saving functions
========================

These functions help saving/loading models in dynet.
"""
import re

import numpy as np
import dynet as dy

_PARAM_NAMES_KEY = "_param_names_"

is_param_id = re.compile(r"^param_([0-9]+)$")
is_param_name = re.compile(r"^(/[^0-9_ ]+(_[0-9]+)*)+\t(param|lookup)$")


def save(pc, filename, compressed=True):
    """Save a ParameterCollection as a `.npz` archive.

    Each parameter is an entry in the archive and its name describes the
    subcollection it lives in.

    Args:
        pc (:py:class:`dynet.ParameterCollection`): Parameter collection to
            save.
        filename (str): Target filename. The ``.npz`` extension will be
            appended to the file name if it is not already there.
        compressed (bool, optional): Compressed ``.npz`` (slower but smaller
            on disk)
    """

    params = {}
    # Retrieve parameters
    for p in pc.parameters_list():
        val = p.as_array()
        header = f"{p.name()}\tparam"
        params[header] = val
    # Retrieve lookup parameters
    for lp in pc.lookup_parameters_list():
        val = lp.as_array()
        header = f"{lp.name()}\tlookup"
        params[header] = val

    param_names = np.asarray(list(params.keys()))
    param_values = [params[name] for name in param_names]
    arrays = {f"param_{i}": value
              for i, value in enumerate(param_values)}
    arrays[_PARAM_NAMES_KEY] = param_names
    # Save all
    if compressed:
        np.savez_compressed(filename, **arrays)
    else:
        np.savez(filename,  **arrays)


def _load_from_npz(filename, ignore_invalid_names=False):
    # Load the npz file
    file_npz = np.load(filename)
    # Get actual parameter names
    if _PARAM_NAMES_KEY not in file_npz.files:
        if not ignore_invalid_names:
            raise ValueError("Wrong format")
        return {}
    actual_names = file_npz[_PARAM_NAMES_KEY]
    # Associate names with values
    params = {}
    param_arrays = [(name, val) for name, val in file_npz.items()
                    if name != _PARAM_NAMES_KEY]
    for name, val in param_arrays:
        # Retrieve the parameter's actual name
        if not is_param_id.match(name):
            if ignore_invalid_names:
                continue
            else:
                raise ValueError(f"Invalid parameter name {name} "
                                 f"in file {filename}")
        param_id = int(name.split("_")[-1])
        actual_name = actual_names[param_id]
        params[actual_name] = val
    # Sort by name
    return params


def load(filename, ignore_invalid_names=False):
    """Load a ParameterCollection from a ``.npz`` file.

    This will recover the subcollection structure.

    Args:
        filename (str): File to load from.
        ignore_invalid_names (bool, optional): Ignore elements with invalid
            parameter names in the ``.npz`` without raising an exception. This
            is useful if for some reason the ``.npz`` contains other arrays.

    Returns:
        :py:class:`dynet.ParameterCollection`: Loaded ParameterCollection
    """
    # All the parameter collections (master collection and subcollections)
    pcs = {"/": dy.ParameterCollection()}
    # Load npz file
    params = _load_from_npz(filename, ignore_invalid_names)
    # Sort by name
    sorted_params = sorted(params.items(), key=lambda x: x[0])
    # Read them in order
    for name, val in sorted_params:
        # Check the name
        if not is_param_name.match(name):
            if ignore_invalid_names:
                continue
            else:
                raise ValueError(
                    f"Invalid parameter name {name} in file {filename}"
                )
        names = name.split("/")
        subpc = "/"
        # Retrieve the subcollection (and create it if necessary)
        for i, pc_name in enumerate(names[:-1]):
            subsubpc = f"{subpc}{pc_name}/"
            if subsubpc not in pcs:
                pcs[subsubpc] = pcs[subpc].add_subcollection(pc_name)
            subpc = subsubpc
        # Parameter type (lookup or not)
        param_type = names[-1].split("\t")[-1]
        # Parameter name (strip collection name and type)
        # This also strips the index (if there are multiple parameters with
        # the same name) but that's not a problem because the parameters are
        # ordered so the correct index will be recovered
        param_name = "\t".join(names[-1].split("\t")[:-1]).split("_")[0]
        # Add to the subcollection
        if param_type is "param":
            pcs[subpc].parameters_from_numpy(val, name=param_name)
        elif param_type is "lookup":
            pcs[subpc].lookup_parameters_from_numpy(val, name=param_name)
    return pcs["/"]


def populate(pc, filename, ignore_shape_mismatch=False):
    """Populate a ParameterCollection from a ``.npz`` file

    Args:
        pc (:py:class:`dynet.ParameterCollection`): Parameter collection to
            populate.
        filename (str): File to populate from.
        ignore_shape_mismatch (bool, optional): Silently ignore shape mismatch
            between the parameter and the value in the ``.npz`` file (just
            don't load the parameter and move on)
    """

    # Load npz
    loaded_params = _load_from_npz(filename, True)
    # Iterate over Parameters in the collection
    for param in pc.parameters_list():
        name = f"{param.name()}\tparam"
        # If the parameter is in the npz, overwrite its value
        if name in loaded_params:
            file_value = loaded_params[name]
            if file_value.shape != param.shape():
                if ignore_shape_mismatch:
                    continue
                else:
                    raise ValueError(
                        f"Shape mismatch for parameter {param.name()}: "
                        f"expected {param.shape()} and found "
                        f"{file_value.shape}"
                    )
            param.set_value(file_value)
    # Iterate over LookupParameters in the collection
    for lookup_param in pc.lookup_parameters_list():
        name = f"{lookup_param.name()}\tlookup"
        # If the parameter is in the npz, overwrite its value
        if name in loaded_params:
            file_value = loaded_params[name]
            if file_value.shape != lookup_param.shape():
                if ignore_shape_mismatch:
                    continue
                else:
                    raise ValueError(
                        f"Shape mismatch for lookup parameter "
                        f"{lookup_param.name()}: expected "
                        f"{lookup_param.shape()} and found {file_value.shape}"
                    )
            for row in range(len(file_value)):
                lookup_param.init_row(row, file_value[row])
