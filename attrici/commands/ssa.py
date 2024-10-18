import argparse
from pathlib import Path


def run(args):
    import xarray as xr

    from attrici.preprocessing import calc_gmt_by_ssa

    input_dataset = xr.open_dataset(args.input)
    gmt = input_dataset[args.variable]
    times = input_dataset["time"]
    ssa_values, ssa_times = calc_gmt_by_ssa(
        gmt, times, window_size=args.window_size, subset=args.subset
    )

    output_dataset = xr.Dataset()
    output_dataset["time"] = ssa_times
    output_dataset["ssa"] = xr.DataArray(ssa_values, dims=["time"])
    output_dataset.to_netcdf(args.output)


def add_parser(subparsers):
    parser = subparsers.add_parser(
        "ssa",
        help="Perform singular spectrum analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="Input file")
    parser.add_argument("output", type=Path, help="Output file")
    parser.add_argument(
        "--variable", type=str, default="tas", help="Variable to process"
    )
    parser.add_argument("--window-size", type=int, default=365, help="Window size")
    parser.add_argument("--subset", type=int, default=10, help="Subset")
    parser.set_defaults(func=run)
