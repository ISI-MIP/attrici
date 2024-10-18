import argparse
from pathlib import Path

import tomllib


class _LoadConfiguration(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        with open(values, "rb") as f:
            config = tomllib.load(f)

        # set required to False for all arguments that are in the configuration,
        # otherwise argparse would raise an error if the argument is not provided
        for action in parser._actions:
            if action.dest in config:
                action.required = False

        # set values from configuration if not already provided on the command line
        for key, value in config.items():
            if getattr(namespace, key, None) is None:
                setattr(namespace, key, value)


def add_config_argument(parser):
    parser.add_argument(
        "--config", type=Path, action=_LoadConfiguration, help="Configuration file"
    )
