"""
Preprocessing functions.
"""

import numpy as np

from attrici.vendored.singularspectrumanalysis import SingularSpectrumAnalysis as SSA


def calc_gmt_by_ssa(gmt, times, window_size=365, subset=10):
    gmt = np.array(np.squeeze(gmt[::subset]), ndmin=2)
    ssa = SSA(window_size).transform(gmt)
    return ssa[0, :], times[::subset]
