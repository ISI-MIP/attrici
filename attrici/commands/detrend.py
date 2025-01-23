import argparse
from datetime import date
from pathlib import Path

from attrici.commands import add_config_argument
from attrici.detrend import Config, detrend


def run(args):
    config_dict = vars(args).copy()
    del config_dict["config"]
    del config_dict["print_config"]
    del config_dict["func"]

    config = Config(**config_dict)

    if args.print_config:
        print(config.to_toml())
        return

    detrend(config)


def iso_date(argument_value):
    """
    Try parsing `argument_value` from an ISO-formatted date into a `date` object.
    Used as an argument type for `--start-date` and `--stop-date`.

    Parameters
    ----------
    argument_value : str
        The string value of the argument

    Returns
    -------
    date
        A date object
    """
    try:
        return date.fromisoformat(argument_value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(e)


def lat_lons(argument_value):
    """
    Try parsing `argument_value` from a semicolon-separated string of lat,lon tuples
    into a list of tuples. Used as an argument type for `--cells`.

    Parameters
    ----------
    argument_value : str
            The string value of the argument

    Returns
    -------
    list of tuple
            A list of lat,lon tuples
    """
    try:
        return [
            (float(lat), float(lon))
            for lat, lon in [pair.split(",") for pair in argument_value.split(";")]
        ]
    except ValueError as e:
        raise argparse.ArgumentTypeError(e)


def add_parser(subparsers):
    parser = subparsers.add_parser(
        "detrend",
        help="Detrend a dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    add_config_argument(parser)

    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print current config as TOML and exit",
    )

    # input
    group = parser.add_argument_group(title="Input")
    group.add_argument(
        "--gmt-file",
        type=Path,
        help="(SSA-smoothed) Global Mean Temperature file",
        required=True,
    )
    group.add_argument(
        "--gmt-variable",
        type=str,
        help="(SSA-smoothed) Global Mean Temperature variable name",
        default=Config.__dataclass_fields__["gmt_variable"].default,
    )
    group.add_argument("--input-file", type=Path, help="Input file", required=True)
    group.add_argument("--mask-file", type=Path, help="Mask file")
    group.add_argument(
        "--variable", type=str, help="Variable to detrend", required=True
    )
    group.add_argument("--trace-file", type=Path, help="Trace file")
    group.add_argument("--fit-only", action="store_true", help="Only fit the model")
    group.add_argument(
        "--cells",
        type=lat_lons,
        help="Semicolon-separated lat,lon tuples to process,"
        " otherwise all cells are processed",
    )

    # output
    group = parser.add_argument_group(title="Output")
    group.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for the results",
        required=True,
    )
    group.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing files"
    )
    group.add_argument("--write-trace", action="store_true", help="Save trace to file")

    # run parameters
    group = parser.add_argument_group(title="Run parameters")
    group.add_argument(
        "--modes",
        type=int,
        default=Config.__dataclass_fields__["modes"].default,
        help="Number of modes for fourier series of model (either one integer, used for"
        " all four series, or four comma-separated integers)",
    )
    group.add_argument(
        "--bootstrap-sample-count",
        type=int,
        default=Config.__dataclass_fields__["bootstrap_sample_count"].default,
        help="Number of bootstrap samples",
    )
    group.add_argument("--progressbar", action="store_true", help="Show progress bar")
    group.add_argument(
        "--report-variables",
        nargs="+",
        default=Config.__dataclass_fields__["report_variables"].default,
        help="Variables to report, e.g. `--report-variables y cfact logp`",
    )
    group.add_argument(
        "--seed",
        type=int,
        default=Config.__dataclass_fields__["seed"].default,
        help="Seed for deterministic randomisation",
    )
    group.add_argument(
        "--solver",
        type=str,
        choices=["pymc5", "scipy", "pymc3"],
        default="pymc5",
        help="Solver library for statistical modelling",
    )
    group.add_argument(
        "--start-date",
        type=iso_date,
        help="Start date for the detrending period",
    )
    group.add_argument(
        "--stop-date",
        type=iso_date,
        help="Stop date for the detrending period",
    )
    group.add_argument(
        "--task-id",
        type=int,
        default=Config.__dataclass_fields__["task_id"].default,
        help="Task ID for parallel processing",
    )
    group.add_argument(
        "--task-count",
        type=int,
        default=Config.__dataclass_fields__["task_count"].default,
        help="Number of tasks for parallel processing",
    )
    group.add_argument(
        "--timeout",
        type=int,
        default=Config.__dataclass_fields__["timeout"].default,
        help="Maximum time in seconds for sampler for a single grid cell",
    )
    group.add_argument(
        "--cache-dir",
        type=Path,
        help="Use cached results from this directory or write new ones",
    )

    parser.set_defaults(func=run)
