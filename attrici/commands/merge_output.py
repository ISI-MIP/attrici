"""
ATTRICI CLI command: merge-output

```
usage: attrici merge-output [--chunksizes CHUNKSIZES] [--mask-file MASK_FILE]
                           directory output_filename

positional arguments:
  directory             Directory containing detrended output timeseries or trace files
  output_filename       Merged output file

options:
  --chunksizes CHUNKSIZES
                        Chunk sizes for dimensions (comma-separated list of
                        dim=chunksize pairs) (default: None)
  --mask-file MASK_FILE
                        Mask file defining output grid; cells with value 1 must
                        have data (default: None)
```
"""

import argparse
from pathlib import Path

import numpy as np
import xarray as xr
from loguru import logger
from netCDF4 import Dataset
from tqdm import tqdm


def _load_mask_grid(mask_path):
    """
    Load mask file and return full-grid lat/lon and the set of (lat, lon) where mask==1.

    Parameters
    ----------
    mask_path : Path
        Path to the mask NetCDF file.

    Returns
    -------
    unique_lats : list
        Sorted latitude values (full grid).
    unique_lons : list
        Sorted longitude values (full grid).
    expected_cells : set of tuple
        Set of (lat, lon) pairs where mask equals 1.
    """
    mask_ds = xr.open_dataset(mask_path)
    if "latitude" in mask_ds.dims and "longitude" in mask_ds.dims:
        mask_ds = mask_ds.rename({"latitude": "lat", "longitude": "lon"})
    if "lat" not in mask_ds.dims or "lon" not in mask_ds.dims:
        raise ValueError(
            "Mask file must have dimensions 'lat' and 'lon' (or 'latitude' and 'longitude')"
        )
    mask_var = mask_ds["mask"]
    if "lat" not in mask_var.dims or "lon" not in mask_var.dims:
        raise ValueError("Mask variable must have dimensions 'lat' and 'lon'")

    unique_lats = sorted(float(x) for x in mask_ds["lat"].values)
    unique_lons = sorted(float(x) for x in mask_ds["lon"].values)

    stacked = mask_var.stack(latlon=("lat", "lon"))
    masked = stacked.where(stacked == 1).dropna("latlon")
    expected_cells = set(
        (float(c[0]), float(c[1])) for c in masked["latlon"].values
    )

    mask_ds.close()
    return unique_lats, unique_lons, expected_cells


def run(args):
    """
    Merge time series data from multiple NetCDF files into a single NetCDF file.

    When a mask file is provided, the output has the same lat/lon dimensions as the
    mask. Every grid cell where the mask is 1 must have a corresponding input file;
    otherwise a ValueError is raised listing the missing cells. Cells where the mask
    is not 1 are left as fill value.

    Parameters
    ----------
    args : argparse.Namespace
        The arguments containing the "directory" of input files, "output_filename",
        and optional "mask_file".
    """
    files = list(args.directory.glob("*/*.nc"))
    if not files:
        raise ValueError(f"No files found: {args.directory}/*/*.nc")

    datasets = [
        xr.open_dataset(fp, chunks="auto", decode_times=False, decode_cf=False)
        for fp in tqdm(files, desc="Loading metadata", leave=False)
    ]

    data_by_cell = {(d.lat.item(), d.lon.item()): d for d in datasets}

    if args.mask_file is not None:
        unique_lats, unique_lons, expected_cells = _load_mask_grid(args.mask_file)
        cells_with_data = set(data_by_cell.keys())
        missing_cells = expected_cells - cells_with_data
        if missing_cells:
            missing_sorted = sorted(missing_cells, key=lambda p: (p[0], p[1]))
            raise ValueError(
                "Mask requires data for the following grid cells, but no input files "
                "were found: "
                + ", ".join(f"({lat:g}, {lon:g})" for lat, lon in missing_sorted)
            )
    else:
        lats = [d.lat.item() for d in datasets]
        lons = [d.lon.item() for d in datasets]
        unique_lats = sorted(set(lats))
        unique_lons = sorted(set(lons))

    d = datasets[0]

    with Dataset(args.output_filename, "w") as nc:
        nc.createDimension("lat", len(unique_lats))
        nc.createVariable("lat", "f4", ("lat",))
        nc["lat"][:] = np.asarray(unique_lats, dtype=np.float32)
        for attr in d["lat"].attrs:
            if not attr.startswith("_"):
                nc["lat"].setncattr(attr, d["lat"].attrs[attr])

        nc.createDimension("lon", len(unique_lons))
        nc.createVariable("lon", "f4", ("lon",))
        nc["lon"][:] = np.asarray(unique_lons, dtype=np.float32)
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
            chunksizes = [c[0] for c in var.chunks] if var.chunks else None
            if args.chunksizes:
                if chunksizes is None:
                    chunksizes = [args.chunksizes.get(dim, None) for dim in var.dims]
                else:
                    for i, dim in enumerate(var.dims):
                        if dim in args.chunksizes:
                            chunksizes[i] = args.chunksizes[dim]
            nc.createVariable(
                var_name,
                var.dtype,
                var.dims,
                chunksizes=chunksizes,
                fill_value=var.attrs.get("_FillValue", None),
                zlib=True,
            )
            for attr in var.attrs:
                if not attr.startswith("_"):
                    nc[var_name].setncattr(attr, var.attrs[attr])

        for (lat, lon), d in tqdm(data_by_cell.items(), desc="Merging data"):
            lat_index = unique_lats.index(lat)
            lon_index = unique_lons.index(lon)
            for var_name in var_names:
                var = d[var_name]
                if "lat" in var.dims or "lon" in var.dims:
                    # we assume lat and lon are in the last two places of dimension
                    # current input (var) has only one of each lat/lon, select it
                    # and put it into the right position in the output (nc)
                    nc[var_name][..., lat_index, lon_index] = var.values[..., 0, 0]
                # Write non-spatial variables once (they are the same for all cells)
                else:
                    nc[var_name][:] = var.values

    for ds in data_by_cell.values():
        ds.close()

    logger.info(f"Saved merged output to {args.output_filename}")


def chunksizes(argument_value):
    """
    Try parsing `argument_value` from a comma-separated string of dimension-chunksize
    pairs (separated by '=')

    Parameters
    ----------
    argument_value : str
        The string value of the argument

    Returns
    -------
    dict
        A dictionary mapping dimension names to chunk sizes
    """
    try:
        return {
            dim: int(chunksize)
            for dim, chunksize in [
                pair.split("=") for pair in argument_value.split(",")
            ]
        }
    except ValueError as e:
        raise argparse.ArgumentTypeError(e)


def add_parser(subparsers):
    """
    Adds the 'merge-output' command to the parser for command-line interface (CLI)
    usage.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        The subparsers action that allows adding subcommands to the main parser.
    """
    parser = subparsers.add_parser(
        "merge-output",
        help="Merge detrended output or trace files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument(
        "--chunksizes",
        type=chunksizes,
        default=None,
        help="Chunk sizes for dimensions (comma-separated list of dim=chunksize pairs)",
    )
    parser.add_argument(
        "--mask-file",
        type=Path,
        default=None,
        help="Mask file defining output grid (same as detrend); output has same dimensions "
        "as mask; cells with value 1 must have data",
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory containing detrended output timeseries or trace files",
    )
    parser.add_argument("output_filename", type=Path, help="Merged output file")
    parser.set_defaults(func=run)
