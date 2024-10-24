import argparse
from pathlib import Path


def run(args):
    import re

    import netCDF4 as nc
    import numpy as np
    import pandas as pd
    import xarray as xr
    from tqdm import tqdm

    r = re.compile(r"ts_lat(-?\d+(\.\d+)?)_lon(-?\d+(\.\d+)?)\.h5")

    input_files = []
    for filename in (args.directory).glob("*/ts_*.h5"):
        m = r.match(filename.name)
        if m:
            lat = float(m.group(1))
            lon = float(m.group(3))
            input_files.append((lat, lon, filename))
    if not input_files:
        raise ValueError("No input files found")
    lats = sorted(set(lat for lat, _, _ in input_files))
    lons = sorted(set(lon for _, lon, _ in input_files))
    times = None

    outfile = xr.Dataset()

    outfile["lat"] = lats
    outfile["lat"].attrs["units"] = "degree_north"
    outfile["lat"].attrs["long_name"] = "latitude"
    outfile["lat"].attrs["standard_name"] = "latitude"
    outfile["lon"] = lons
    outfile["lon"].attrs["units"] = "degree_east"
    outfile["lon"].attrs["long_name"] = "longitude"
    outfile["lon"].attrs["standard_name"] = "longitude"

    lat, lon, filename = input_files[0]
    store = pd.HDFStore(filename, "r")
    df_name = f"lat_{lat}_lon_{lon}"
    d = store[df_name]
    if "ds" not in d:
        raise ValueError("No 'ds' column found")
    times = d["ds"].to_numpy()
    outfile["time"] = times
    for name, value in store.get_storer(df_name).attrs.metadata.items():
        outfile.attrs[name] = value
    store.close()
    outfile.to_netcdf(
        args.output_filename,
        encoding={
            "time": {
                "dtype": "int32",
            }
        },
    )

    outfile = nc.Dataset(args.output_filename, "a")
    var = outfile.createVariable(
        args.output_variable,
        "f4",
        ("time", "lat", "lon"),
        chunksizes=(len(times), 1, 1),
        zlib=True,
        complevel=7,
    )
    try:
        for lat, lon, filename in tqdm(input_files):
            d = pd.read_hdf(filename)
            if np.any(d["ds"].to_numpy() != times):
                raise ValueError("Time mismatch")
            lat_index = lats.index(lat)
            lon_index = lons.index(lon)
            var[:, lat_index, lon_index] = d[args.variable].to_numpy()
    except KeyError as e:
        raise ValueError(f"Column {e} not found in input files")


def add_parser(subparsers):
    parser = subparsers.add_parser(
        "merge-output",
        help="Merge detrended output",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "directory", type=Path, help="Directory containing detrended output timeseries"
    )
    parser.add_argument("variable", type=str, help="Variable to merge")
    parser.add_argument("output_variable", type=str, help="Variable name in output")
    parser.add_argument("output_filename", type=Path, help="Merged output file")
    parser.set_defaults(func=run)
