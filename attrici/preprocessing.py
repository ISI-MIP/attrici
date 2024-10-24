"""
Preprocessing functions.
"""

# TODO: this script has special dependency: pyts
# pyts needs numba and scikit-learn
# they are complicated to handle with the others.
# Longer-term, we should use standard packages.

import numpy as np
from pyts.decomposition import SingularSpectrumAnalysis as SSA


def calc_gmt_by_ssa(gmt, times, window_size=365, subset=10):
    gmt = np.array(np.squeeze(gmt[::subset]), ndmin=2)
    ssa = SSA(window_size).fit_transform(gmt)
    return ssa[0, 0], times[::subset]
