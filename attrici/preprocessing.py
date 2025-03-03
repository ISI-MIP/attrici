"""
Preprocessing functions.

This module for now contains only a function to derive a smoothed global mean
temperature (GMT) using Singular Spectrum Analysis (SSA).
"""

import numpy as np

from attrici.vendored.singularspectrumanalysis import SingularSpectrumAnalysis as SSA


def calc_gmt_by_ssa(gmt, times, window_size=365, subset=10):
    """
    Calculate a smoothed version (i.e. without periodic oscillations) of the global
    mean temperature (GMT) using Singular Spectrum Analysis (SSA).

    Parameters
    ----------
    gmt : array_like
        Array of global mean temperature values.
    times : array_like
        Array of corresponding time values.
    window_size : int, optional
        The window size for SSA, by default 365.
    subset : int, optional
        The step size for subsetting the input arrays, by default 10.

    Returns
    -------
    tuple
        A tuple containing:
        - ssa[0, :] : ndarray
            The first reconstructed component from SSA.
        - times[::subset] : ndarray
            The subset of time values corresponding to the SSA result.
    """
    gmt = np.array(np.squeeze(gmt[::subset]), ndmin=2)
    ssa = SSA(window_size).transform(gmt)
    return ssa[0, :], times[::subset]
