"""
Utility functions such as for timing, data provenance, and oscillation calculation.
"""

import importlib.metadata
import sys
import time
from datetime import datetime

import numpy as np
import xarray as xr
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


def collect_windows(data, window_size):
    """
    Collect rolling windows of a given size around each day of the year.

    Parameters
    ----------
    data : xarray.DataArray
        The timeseries to collect windows from.
    window_size : int
        The size of the window around each day of the year. Should be an odd number.

    Returns
    -------
    xarray.DataArray
        Data array with dimensions `dayofyear` and `index_in_window` containing, for
        each day of the year, the joint windows around that day for all years in the
        original timeseries.
    """

    # Collect indices for each day-of-year for all years
    dayofyear_indices = data.time.groupby("time.dayofyear").groups

    # We want to model data for each day of the year. In case of leap years we also need
    # to model day 366, for that day we use the indices of days following the 365th day
    # of each year, i.e. the last day of each leap year and the first day after each
    # non-leap year
    dayofyear_indices[366] = [i + 1 for i in dayofyear_indices[365]]

    # To have complete windows in the beginning and end of the timeseries we extend the
    # series by mirroring half a window around the edges. One additional full window is
    # kept at the end of the timeseries (hence `half_window_size - 2`) because otherwise
    # the last index in dayofyear_indices[366] might be outside the timeseries.
    half_window_size = window_size // 2
    extended_data = xr.concat(
        [
            data.isel(time=list(reversed(range(1, half_window_size + 1)))),
            data,
            data.isel(
                time=list(
                    reversed(range(len(data) - half_window_size - 2, len(data) - 1))
                )
            ),
        ],
        dim="time",
    )

    # For each day we want a new array around it with size `window_size`
    rolling_day_windows = (
        extended_data.rolling(time=window_size, center=True)
        .construct("index_in_day_window")
        .isel(time=slice(half_window_size, len(extended_data) - half_window_size))
    )

    # Now join all windows for each day of year
    result = xr.concat(
        [
            rolling_day_windows.isel(time=dayofyear_indices[k])
            .stack(index_in_window=("time", "index_in_day_window"))
            .reset_index("index_in_window", drop=True)
            for k in range(1, 367)
        ],
        dim="dayofyear",
    )
    return result.assign_coords(
        index_in_window=range(result.sizes["index_in_window"]), dayofyear=range(1, 367)
    ).stack(time=("dayofyear", "index_in_window"), create_index=False)
