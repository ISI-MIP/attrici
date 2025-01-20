import argparse
from pathlib import Path

import xarray as xr
from loguru import logger


def run(args):
    ds = xr.open_mfdataset(
        str(args.directory / "*/*.nc"),
        parallel=True,
        engine="h5netcdf",
        # for parallel=True, only engine="h5netcdf" seems to work for now
    )
    ds.to_netcdf(args.output_filename)
    logger.info(f"Saved merged output to {args.output_filename}")


def add_parser(subparsers):
    parser = subparsers.add_parser(
        "merge-output",
        help="Merge detrended output or trace files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory containing detrended output timeseries or trace files",
    )
    parser.add_argument("output_filename", type=Path, help="Merged output file")
    parser.set_defaults(func=run)
