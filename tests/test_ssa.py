# TODO check whether still needed once updated to latest numpy version
import netCDF4  # noqa Used to suppress warning https://github.com/pydata/xarray/issues/7259
import numpy as np
import pytest
import xarray as xr

from attrici import preprocessing
from attrici.ssa import ssa


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


@pytest.mark.slow
def test_ssa():
    INPUT_FILE = "tests/data/20crv3-era5_gmt_raw.nc"
    OUTPUT_FILE = "tests/data/output/20crv3-era5_gmt_ssa.nc"
    REFERENCE_FILE = "tests/data/20CRv3-ERA5_germany_ssa_gmt.nc"

    ssa(
        input=INPUT_FILE,
        variable="tas",
        window_size=365,
        subset=10,
        output=OUTPUT_FILE,
    )

    desired = xr.load_dataset(REFERENCE_FILE)
    actual = xr.load_dataset(OUTPUT_FILE)

    np.testing.assert_allclose(actual.tas, desired.tas, rtol=1e-05, atol=1e-06)
