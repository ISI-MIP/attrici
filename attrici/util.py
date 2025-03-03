"""
Utility functions such as for timing, data provenance, and oscillation calculation.
"""

import importlib.metadata
import sys
import time
from datetime import datetime

import numpy as np
from loguru import logger

import attrici


def get_data_provenance_metadata(**other_metadata):
    """
    Assemble metadata for data provenance

    Returns
    -------
    dictionary
        Dictionary with provenance information
    """
    res = {
        "attrici_version": attrici.__version__,
        "attrici_packages": "\n".join(
            sorted(
                {
                    f"{dist.name}=={dist.version}"
                    for dist in importlib.metadata.distributions()
                },
                key=str.casefold,
            )
        ),
        "attrici_python_version": sys.version,
        "created_at": datetime.now().isoformat(),
    }
    res.update(other_metadata)
    return res


def timeit(func):
    """
    Decorator to time a function during logging.

    Parameters
    ----------
    func : callable
        The function to be wrapped and timed.

    Returns
    -------
    callable
        The wrapped function that logs its execution time.

    See Also
    --------
    `loguru documentation <https://loguru.readthedocs.io/en/stable/resources/recipes.html#logging-entry-and-exit-of-functions-with-a-decorator>`__
    """

    def _wrapped(*args, **kwargs):
        """
        Wrapped function that logs the execution time.

        Returns
        -------
        object
            The result of the wrapped function.
        """
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.info(
            "Function '{}' executed in {:.1f} minutes",
            func.__name__,
            (end - start) / 60,
        )
        return result

    return _wrapped


def calc_oscillations(t, modes):
    """
    Calculate oscillations based on time and modes.

    Parameters
    ----------
    t : xarray.DataArray
        Array of time values.
    modes : int
        Number of modes.

    Returns
    -------
    numpy.ndarray
        Array of oscillation values.
    """
    t_scaled = (t - t.min()) / (np.timedelta64(365, "D") + np.timedelta64(6, "h"))
    x = (2 * np.pi * (np.arange(modes) + 1)) * t_scaled.values[:, None]
    return np.concatenate((np.cos(x), np.sin(x)), axis=1)
