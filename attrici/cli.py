"""
ATTRICI Command Line Interface (CLI)

Output from `attrici --help`:
```
usage: attrici [-h] [--version] {detrend, merge-output, merge-traces, \\
    preprocess-tas, postprocess-tas, ssa} ...

Calculates counterfactual climate data from past datasets.

positional arguments:
  {detrend,merge-output,merge-traces,preprocess-tas,postprocess-tas,ssa}
                        Action to perform
    detrend             Detrend a dataset
    merge-output        Merge detrended output
    merge-traces        Merge traces from detrend run
    preprocess-tas      Derive tasrange and tasskew from tas, tasmin, and tasmax
    postprocess-tas     Derive tasmin and tasmax from tas, tasrange, and tasskew
    ssa                 Perform singular spectrum analysis

options:
  -h, --help            show this help message and exit
  --version             show program's version number and exit

Source code: https://github.com/ISI-MIP/attrici
Method: https://doi.org/10.5194/gmd-14-5269-2021
```

See `attrici.commands` for sub-commands.
"""

import argparse
import sys

from attrici import __version__
from attrici.commands import (
    derive_huss,
    detrend,
    merge_output,
    postprocess_tas,
    preprocess_tas,
    ssa,
)


def main():
    """
    Main entry point for the attrici command-line tool.

    This function sets up the command-line interface using `argparse`.
    """
    parser = argparse.ArgumentParser(
        prog="attrici",
        description="Calculates counterfactual climate data from past datasets.",
        epilog="""
        Source code: https://github.com/ISI-MIP/attrici

        Method: https://doi.org/10.5194/gmd-14-5269-2021
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )

    # global arguments
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )

    subparsers = parser.add_subparsers(help="Action to perform", required=True)

    # commands
    derive_huss.add_parser(subparsers)
    detrend.add_parser(subparsers)
    merge_output.add_parser(subparsers)
    postprocess_tas.add_parser(subparsers)
    preprocess_tas.add_parser(subparsers)
    ssa.add_parser(subparsers)

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()
    args.func(args)
