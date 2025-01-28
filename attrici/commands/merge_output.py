import argparse
from pathlib import Path

import xarray as xr
from loguru import logger
from netCDF4 import Dataset
from tqdm import tqdm


def run(args):
    files = list(args.directory.glob("*/*.nc"))
    if not files:
        raise ValueError(f"No files found: {args.directory}/*/*.nc")

    datasets = [
        xr.open_dataset(fp, chunks="auto", decode_times=False, decode_cf=False)
        for fp in tqdm(files, desc="Loading metadata", leave=False)
    ]

    lats = [d.lat.item() for d in datasets]
    lons = [d.lon.item() for d in datasets]
    unique_lats = sorted(set(lats))
    unique_lons = sorted(set(lons))

    with Dataset(args.output_filename, "w") as nc:
        d = datasets[0]

        nc.createDimension("lat", len(unique_lats))
        nc.createVariable("lat", "f4", ("lat",))
        nc["lat"][:] = unique_lats
        for attr in d["lat"].attrs:
            if not attr.startswith("_"):
                nc["lat"].setncattr(attr, d["lat"].attrs[attr])

        nc.createDimension("lon", len(unique_lons))
        nc.createVariable("lon", "f4", ("lon",))
        nc["lon"][:] = unique_lons
        for attr in d["lon"].attrs:
            if not attr.startswith("_"):
                nc["lon"].setncattr(attr, d["lon"].attrs[attr])

        for attr in d.attrs:
            nc.setncattr(attr, d.attrs[attr])

        for dim in set(d.dims) - {"lat", "lon"}:
            nc.createDimension(dim, d[dim].size)
            if dim in d:
                nc.createVariable(dim, d[dim].dtype, (dim,))
                nc[dim][:] = d[dim].values
                for attr in d[dim].attrs:
                    if not attr.startswith("_"):
                        nc[dim].setncattr(attr, d[dim].attrs[attr])

        var_names = d.data_vars.keys()
        for var_name in set(var_names) - {"lat", "lon"}:
            var = d[var_name]
            # we assume lat and lon are in the last two places of dimension
            if ("lat" in var.dims or "lon" in var.dims) and (
                var.dims[-2] != "lat" or var.dims[-1] != "lon"
            ):
                raise ValueError(
                    f"Variable {var_name} has lat/lon dimensions in the wrong order"
                )
            nc.createVariable(
                var_name,
                var.dtype,
                var.dims,
                chunksizes=[c[0] for c in var.chunks],
                fill_value=var.encoding.get("_FillValue", None),
                zlib=True,
            )
            for attr in var.attrs:
                if not attr.startswith("_"):
                    nc[var_name].setncattr(attr, var.attrs[attr])

        for i in tqdm(range(len(datasets)), desc="Merging data"):
            d = datasets.pop(0)
            for var_name in var_names:
                var = d[var_name]
                if "lat" in var.dims or "lon" in var.dims:
                    lat_index = unique_lats.index(d.lat.item())
                    lon_index = unique_lons.index(d.lon.item())
                    # we assume lat and lon are in the last two places of dimension
                    # current input (var) has only one of each lat/lon, select it
                    # and put it into the right position in the output (nc)
                    nc[var_name][..., lat_index, lon_index] = var.values[..., 0, 0]
                else:
                    nc[var_name][:] = var.values

    logger.info(f"Saved merged output to {args.output_filename}")


def add_parser(subparsers):
    parser = subparsers.add_parser(
        "merge-output",
        help="Merge detrended output or trace files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory containing detrended output timeseries or trace files",
    )
    parser.add_argument("output_filename", type=Path, help="Merged output file")
    parser.set_defaults(func=run)
