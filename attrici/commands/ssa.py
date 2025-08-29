"""
ATTRICI CLI command: ssa

```
usage: attrici ssa [--variable VARIABLE] [--window-size WINDOW_SIZE] \\
  [--subset SUBSET] input output

positional arguments:
  input                 Input file
  output                Output file

options:
  --variable VARIABLE   Variable to process (default: tas)
  --window-size WINDOW_SIZE
                        Window size (default: 365)
  --subset SUBSET       Subset (default: 10)
```

See `attrici.vendored.singularspectrumanalysis` for the SSA implementation.
"""

import argparse
from pathlib import Path

from attrici.ssa import ssa


def run(args):
    """
    Run Singular Spectrum Analysis (SSA) on the input data.

    Parameters
    ----------
    args : argparse.Namespace
        Values for input and output file, variable name, window size, and subset.
    """
    ssa(args.input, args.variable, args.window_size, args.subset, args.output)


def add_parser(subparsers):
    """
    Adds the 'ssa' command to the parser for command-line interface (CLI) usage.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        The subparsers action that allows adding subcommands to the main parser.
    """
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
