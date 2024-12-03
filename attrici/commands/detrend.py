import argparse
from datetime import date
from pathlib import Path

from attrici.commands import add_config_argument


def run(args):
    from attrici.detrend import Config, detrend

    config_dict = vars(args).copy()
    del config_dict["config"]
    del config_dict["print_config"]
    del config_dict["func"]

    config = Config(**config_dict)

    if args.print_config:
        print(config.to_toml())
        return

    detrend(
        config,
        progressbar=args.progressbar,
        use_cache=args.use_cache,
        overwrite=args.overwrite,
    )


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


def add_parser(subparsers):
    parser = subparsers.add_parser(
        "detrend",
        help="Detrend a dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
    group.add_argument("--input-file", type=Path, help="Input file", required=True)
    group.add_argument("--mask-file", type=Path, help="Mask file")
    group.add_argument(
        "--variable", type=str, help="Variable to detrend", required=True
    )

    # output
    group = parser.add_argument_group(title="Output")
    group.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Output directory for the results",
    )
    group.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing files"
    )

    # run parameters
    group = parser.add_argument_group(title="Run parameters")
    group.add_argument(
        "--modes",
        type=int,
        default=4,
        help="Number of modes for fourier series of model (either one integer, used for"
        " all four series, or four comma-separated integers)",
    )
    group.add_argument("--progressbar", action="store_true", help="Show progress bar")
    group.add_argument(
        "--report-variables",
        nargs="+",
        default=["all"],
        help="Variables to report, e.g. `--report-variables ds y cfact logp`",
    )
    group.add_argument(
        "--seed", type=int, default=0, help="Seed for deterministic randomisation"
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
        "--task-id", type=int, default=0, help="Task ID for parallel processing"
    )
    group.add_argument(
        "--task-count",
        type=int,
        default=1,
        help="Number of tasks for parallel processing",
    )
    group.add_argument(
        "--timeout",
        type=int,
        default=60 * 60,
        help="Maximum time in seconds for sampler for a single grid cell",
    )
    group.add_argument(
        "--use-cache", action="store_true", help="Use cached results and write new ones"
    )

    # model parameters
    group = parser.add_argument_group(title="Model parameters")
    group.add_argument(
        "--tune",
        type=int,
        default=500,
        help="Number of draws to tune model",
    )
    group.add_argument(
        "--draws",
        type=int,
        default=1000,
        help="Number of sampling draws per chain",
    )
    group.add_argument(
        "--chains",
        type=int,
        default=2,
        help="Number of chains to calculate (min 2 to check for convergence)",
    )

    parser.set_defaults(func=run)
