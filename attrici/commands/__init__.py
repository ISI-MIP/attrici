"""
ATTRICI Command Line Interface (CLI) sub-commands.

See also `attrici.cli`.
"""

import argparse
import tomllib
from pathlib import Path


class _LoadConfiguration(argparse.Action):
    """
    Custom argparse action to load configuration from a TOML file.

    The TOML configuration is loaded and the argparse namespace is updated with
    the values from the file.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        """
        Load configuration from a TOML file and update the argparse namespace.

        Parameters
        ----------
        parser : argparse.ArgumentParser
            The argument parser instance.
        namespace : argparse.Namespace
            The namespace object that will be updated with the configuration values.
        values : str or pathlib.Path
            The path to the TOML configuration file.
        option_string : str, optional
            The option string that was used to invoke this action, by default None
        """
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
    """
    Adds a '--config' argument to the argparse parser for specifying a configuration
    file.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to which the '--config' argument will be added.
    """
    parser.add_argument(
        "--config", type=Path, action=_LoadConfiguration, help="Configuration file"
    )
