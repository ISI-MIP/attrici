"""
ATTRICI CLI command: preprocess-tas

```
usage: attrici preprocess-tas tas tasmin tasmax tasrange tasskew

positional arguments:
  tas         Input tas file
  tasmin      Input tasmin file
  tasmax      Input tasmax file
  tasrange    Output tasrange file
  tasskew     Output tasskew file
```
"""

import argparse
import subprocess
import sys
from pathlib import Path

from loguru import logger


def run(args):
    """
    Preprocesses temperature data by deriving 'tasrange' and 'tasskew' from 'tas',
    'tasmin', and 'tasmax' using [CDO (Climate Data
    Operators)](https://code.mpimet.mpg.de/projects/cdo). Input files are processed with
    subprocess calls to `cdo` which needs to be installed and on the path.

    Parameters
    ----------
    args : argparse.Namespace
        The arguments containing input and output file paths for 'tas', 'tasmin',
        'tasmax', 'tasrange', and 'tasskew'.
    """

    try:
        cdo_version = (
            subprocess.check_output(["cdo", "-V"]).decode("utf-8").split("\n")[0]
        )
    except FileNotFoundError:
        logger.error("CDO not found. Please install CDO.")
        sys.exit(1)
    else:
        logger.info("Using {}", cdo_version)

    cdo_command = ["cdo", "-O", "-f", "nc4", "-z", "zip"]

    try:
        subprocess.check_call(
            cdo_command
            + [
                "-chname,tasmax,tasrange",
                "-sub",
                args.tasmax,
                args.tasmin,
                args.tasrange,
            ]
        )
        subprocess.check_call(
            cdo_command
            + [
                "-chname,tas,tasskew",
                "-div",
                "-sub",
                args.tas,
                args.tasmin,
                args.tasrange,
                args.tasskew,
            ]
        )
    except subprocess.CalledProcessError as e:
        logger.error("CDO failed with exit code {}", e.returncode)
        sys.exit(e.returncode)


def add_parser(subparsers):
    """
    Adds the 'preprocess-tas' command to th parser for command-line interface (CLI)
    usage.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        The subparsers action that allows adding subcommands to the main parser.
    """
    parser = subparsers.add_parser(
        "preprocess-tas",
        help="Derive tasrange and tasskew from tas, tasmin, and tasmax",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("tas", type=Path, help="Input tas file")
    parser.add_argument("tasmin", type=Path, help="Input tasmin file")
    parser.add_argument("tasmax", type=Path, help="Input tasmax file")
    parser.add_argument("tasrange", type=Path, help="Output tasrange file")
    parser.add_argument("tasskew", type=Path, help="Output tasskew file")
    parser.set_defaults(func=run)
