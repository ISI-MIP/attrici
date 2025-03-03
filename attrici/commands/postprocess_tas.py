"""
ATTRICI CLI command: postprocess-tas

```
usage: attrici postprocess-tas tas tasrange tasskew tasmin tasmax

positional arguments:
  tas         Input tas file
  tasrange    Input tasrange file
  tasskew     Input tasskew file
  tasmin      Output tasmin file
  tasmax      Output tasmax file
```
"""

import argparse
import subprocess
import sys
from pathlib import Path

from loguru import logger


def run(args):
    """
    Post-processes temperature data by deriving 'tasmin' and 'tasmax' from 'tas',
    'tasrange', and 'tasskew' using [CDO (Climate Data
    Operators)](https://code.mpimet.mpg.de/projects/cdo). Input files are processed with
    subprocess calls to `cdo` which needs to be installed and on the path.

    Parameters
    ----------
    args : argparse.Namespace
        The arguments containing input and output file paths for 'tas', 'tasrange',
        'tasskew', 'tasmin', and 'tasmax'.
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
                "-chname,tas,tasmin",
                "-sub",
                args.tas,
                "-mul",
                args.tasskew,
                args.tasrange,
                args.tasmin,
            ]
        )
        subprocess.check_call(
            cdo_command
            + [
                "-chname,tasmin,tasmax",
                "-add",
                args.tasmin,
                args.tasrange,
                args.tasmax,
            ]
        )
    except subprocess.CalledProcessError as e:
        logger.error("CDO failed with exit code {}", e.returncode)
        sys.exit(e.returncode)


def add_parser(subparsers):
    """
    Adds the 'postprocess-tas' command to the parser for command-line interface (CLI)
    usage.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        The subparsers action that allows adding subcommands to the main parser.
    """
    parser = subparsers.add_parser(
        "postprocess-tas",
        help="Derive tasmin and tasmax from tas, tasrange, and tasskew",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("tas", type=Path, help="Input tas file")
    parser.add_argument("tasrange", type=Path, help="Input tasrange file")
    parser.add_argument("tasskew", type=Path, help="Input tasskew file")
    parser.add_argument("tasmin", type=Path, help="Output tasmin file")
    parser.add_argument("tasmax", type=Path, help="Output tasmax file")
    parser.set_defaults(func=run)
