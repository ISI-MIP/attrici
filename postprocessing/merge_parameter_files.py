import glob
import pickle
import re
from pathlib import Path

import numpy as np
import settings as s
import xarray as xr


def flatten_dict(d):
    flat_dict = {}
    for key, value in d.items():
        # Check if value is a numpy array and has zero dimensions
        if isinstance(value, np.ndarray) and value.ndim == 0:
            flat_dict[key] = float(value.item())
        elif isinstance(value, (list, tuple, np.ndarray)):
            for i, v in enumerate(value):
                new_key = f"{key}_{i}"
                flat_dict[new_key] = float(v)
        else:
            flat_dict[key] = float(value)
    return flat_dict


def unflatten_dict(flat_dict):
    original_dict = {}
    for key, value in flat_dict.items():
        parts = key.split("_")
        if parts[-1].isdigit():
            base_key = "_".join(parts[:-1])
            index = int(parts[-1])
            if base_key not in original_dict:
                original_dict[base_key] = []
            # Extend the list if necessary
            while len(original_dict[base_key]) <= index:
                original_dict[base_key].append(0.0)
            original_dict[base_key][index] = value
        else:
            original_dict[key] = value
    # Convert lists to numpy arrays and handle scalar values
    for key, value in original_dict.items():
        if isinstance(value, list):
            original_dict[key] = np.array(value)
        elif isinstance(value, float):
            original_dict[key] = np.array(value)
    return original_dict


def get_float_from_string(file_name):
    """
    Extract floats from foldernames or filenames
    """
    floats_in_string = re.findall(r"[-+]?(?:\d*\.*\d+)", file_name)
    if len(floats_in_string) != 1:
        raise ValueError("there is more than one float in this string")
    return float(floats_in_string[0])


def merge_parameters():
    trace_dir = s.output_dir / "traces" / s.variable
    # load the landmask dataset and extract the lat and lon values
    landmask_ds = xr.open_dataset(s.input_dir / s.landsea_file)

    parameter_datasets = []
    for file in sorted(trace_dir.glob("**/lon*")):
        lat = get_float_from_string(file.parent.name)
        lon = get_float_from_string(file.name.split("lon")[-1])

        with open(file, "rb") as trace:
            free_params_dict = flatten_dict(pickle.load(trace))
        ds = xr.Dataset(
            {
                key: xr.DataArray(
                    data=[[value]],
                    coords={"lat": [lat], "lon": [lon]},
                    dims=["lat", "lon"],
                )
                for key, value in free_params_dict.items()
            }
        )
        parameter_datasets.append(ds)
    combined_ds = xr.combine_by_coords(parameter_datasets, compat="no_conflicts")
    final_ds = xr.combine_by_coords([landmask_ds, combined_ds], compat="no_conflicts")
    merged_parameter_file = f"{s.source_file.split('.')[0]}_parameters.nc"
    final_ds.to_netcdf(s.output_dir / merged_parameter_file)
    print(f"saved parameters to {merged_parameter_file}")
    return final_ds


if __name__ == "__main__":
    merge_parameters()
