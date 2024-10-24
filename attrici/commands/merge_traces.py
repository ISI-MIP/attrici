import argparse
from pathlib import Path


def run(args):
    import pickle
    import re

    import numpy as np
    import xarray as xr
    from tqdm import tqdm

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

    r = re.compile(r"traces_lat(-?\d+(\.\d+)?)_lon(-?\d+(\.\d+)?)\.pkl")

    input_files = []
    for filename in (args.directory).glob("*/traces_*.pkl"):
        m = r.match(filename.name)
        if m:
            lat = float(m.group(1))
            lon = float(m.group(3))
            input_files.append((lat, lon, filename))
    if not input_files:
        raise ValueError("No input files found")
    lats = sorted(set(lat for lat, _, _ in input_files))
    lons = sorted(set(lon for _, lon, _ in input_files))

    outfile = xr.Dataset()

    outfile["lat"] = lats
    outfile["lat"].attrs["units"] = "degree_north"
    outfile["lat"].attrs["long_name"] = "latitude"
    outfile["lat"].attrs["standard_name"] = "latitude"
    outfile["lon"] = lons
    outfile["lon"].attrs["units"] = "degree_east"
    outfile["lon"].attrs["long_name"] = "longitude"
    outfile["lon"].attrs["standard_name"] = "longitude"

    results = {}
    for lat, lon, filename in tqdm(input_files):
        lat_index = lats.index(lat)
        lon_index = lons.index(lon)
        with open(filename, "rb") as trace:
            free_params_dict = flatten_dict(pickle.load(trace))
        for key, value in free_params_dict.items():
            if key not in results:
                results[key] = np.full((len(lats), len(lons)), np.nan)
            results[key][lat_index, lon_index] = value

    for key, values in results.items():
        outfile[key] = (("lat", "lon"), values)

    outfile.to_netcdf(args.output_filename)


def add_parser(subparsers):
    parser = subparsers.add_parser(
        "merge-traces",
        help="Merge traces from detrend run",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("directory", type=Path, help="Directory containing traces")
    parser.add_argument("output_filename", type=Path, help="Merged output file")
    parser.set_defaults(func=run)
