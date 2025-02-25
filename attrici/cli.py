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
    parser = argparse.ArgumentParser(
        prog="ATTRICI",
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
