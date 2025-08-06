import argparse
from pathlib import Path

from attrici.ssa import ssa


def run(args):
    ssa(args.input, args.variable, args.window_size, args.subset, args.output)


def add_parser(subparsers):
    parser = subparsers.add_parser(
        "ssa",
        help="Perform singular spectrum analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("input", type=Path, help="Input file")
    parser.add_argument("output", type=Path, help="Output file")
    parser.add_argument(
        "--variable", type=str, default="tas", help="Variable to process"
    )
    parser.add_argument("--window-size", type=int, default=365, help="Window size")
    parser.add_argument("--subset", type=int, default=10, help="Subset")
    parser.set_defaults(func=run)
