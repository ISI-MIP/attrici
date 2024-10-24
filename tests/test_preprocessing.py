import numpy as np

from attrici import preprocessing


def test_calc_gmt_by_ssa():
    times = np.arange(0, 365 * 10)

    # Create test data as a series of normally distributed random numbers around a
    # linear GMT trend
    trend = np.linspace(15, 18, len(times))
    gmt = np.random.normal(trend, 2, len(times))

    subset = 10
    ssa, ssa_times = preprocessing.calc_gmt_by_ssa(
        gmt, times, window_size=365, subset=subset
    )

    assert ssa.shape == (len(times[::subset]),)
    assert ssa_times.shape == (len(times[::subset]),)
