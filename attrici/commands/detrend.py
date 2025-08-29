"""
ATTRICI CLI command: detrend

```
usage: attrici detrend [-h] [--config CONFIG] [--print-config] --gmt-file GMT_FILE
                       [--gmt-variable GMT_VARIABLE] --input-file INPUT_FILE
                       [--mask-file MASK_FILE] --variable VARIABLE
                       [--trace-file TRACE_FILE] [--fit-only] [--cells CELLS]
                       --output-dir OUTPUT_DIR [--overwrite] [--write-trace]
                       [--modes MODES] [--window-size WINDOW_SIZE]
                       [--bootstrap-sample-count BOOTSTRAP_SAMPLE_COUNT]
                       [--full-extrapolation] [--progressbar]
                       [--report-variables REPORT_VARIABLES [REPORT_VARIABLES ...]]
                       [--seed SEED] [--solver {pymc5,scipy,pymc3}]
                       [--start-date START_DATE] [--stop-date STOP_DATE]
                       [--task-id TASK_ID] [--task-count TASK_COUNT]
                       [--compile-timeout COMPILE_TIMEOUT] [--timeout TIMEOUT]
                       [--cache-dir CACHE_DIR]

options:
  -h, --help            show this help message and exit
  --config CONFIG       Configuration file (default: None)
  --print-config        Print current config as TOML and exit (default: False)

Input:
  --gmt-file GMT_FILE   (SSA-smoothed) Global Mean Temperature file (default: None)
  --gmt-variable GMT_VARIABLE
                        (SSA-smoothed) Global Mean Temperature variable name
                        (default: tas)
  --input-file INPUT_FILE
                        Input file (default: None)
  --mask-file MASK_FILE
                        Mask file (default: None)
  --variable VARIABLE   Variable to detrend (default: None)
  --trace-file TRACE_FILE
                        Trace file (default: None)
  --fit-only            Only fit the model (default: False)
  --cells CELLS         Semicolon-separated lat,lon tuples to process, otherwise
                        all cells are processed (default: None)

Output:
  --output-dir OUTPUT_DIR
                        Output directory for the results (default: None)
  --overwrite           Overwrite existing files (default: False)
  --write-trace         Save trace to file (default: False)

Run parameters:
  --modes MODES         Number of modes for fourier series of model (defaults to 4
                        if not set) (default: None)
  --window-size WINDOW_SIZE
                        Size of the window around each day of the year. Must be an
                        odd number. (default: None)
  --bootstrap-sample-count BOOTSTRAP_SAMPLE_COUNT
                        Number of bootstrap samples (default: 0)
  --full-extrapolation  Extrapolate few missing days of GMT instead of stretching
                        it to the full time series (default: False)
  --progressbar         Show progress bar (default: False)
  --report-variables REPORT_VARIABLES [REPORT_VARIABLES ...]
                        Variables to report, e.g. `--report-variables y cfact logp`
                        (default: ('all',))
  --seed SEED           Seed for deterministic randomisation (default: 0)
  --solver {pymc5,scipy,pymc3}
                        Solver library for statistical modelling (default: pymc5)
  --start-date START_DATE
                        Start date for the detrending period (default: None)
  --stop-date STOP_DATE
                        Stop date for the detrending period (default: None)
  --task-id TASK_ID     Task ID for parallel processing (default: 0)
  --task-count TASK_COUNT
                        Number of tasks for parallel processing (default: 1)
  --compile-timeout COMPILE_TIMEOUT
                        Timeout for PyMC5 model compilation in s (default: 600)
  --timeout TIMEOUT     Maximum time in seconds for sampler for a single grid cell
                        (default: 3600)
  --cache-dir CACHE_DIR
                        Use cached results from this directory or write new ones
                        (default: None)
```
"""

import argparse
from datetime import date
from pathlib import Path

from attrici.commands import add_config_argument
from attrici.detrend import Config, detrend

MODES_DEFAULT = 4


def run(args):
    """
    Run detrend command.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments
    """
    config_dict = vars(args).copy()
    del config_dict["config"]
    del config_dict["print_config"]
    del config_dict["func"]

    config = Config(**config_dict)

    if args.modes is None and args.window_size is None:
        config.modes = MODES_DEFAULT

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
    """
    Add an argparse parser for the 'detrend' command.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        The subparsers object to which the new parser will be added.
    """
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
        help=f"Number of modes for fourier series of model (defaults to {MODES_DEFAULT}"
        " if not set)",
    )
    group.add_argument(
        "--window-size",
        type=int,
        default=Config.__dataclass_fields__["window_size"].default,
        help="Size of the window around each day of the year. Must be an odd number.",
    )
    group.add_argument(
        "--bootstrap-sample-count",
        type=int,
        default=Config.__dataclass_fields__["bootstrap_sample_count"].default,
        help="Number of bootstrap samples",
    )
    group.add_argument(
        "--full-extrapolation",
        action="store_true",
        help="Extrapolate few missing days of GMT instead of stretching it to the full"
        " time series",
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
        "--compile-timeout",
        type=int,
        default=Config.__dataclass_fields__["compile_timeout"].default,
        help="Timeout for PyMC5 model compilation in s",
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
