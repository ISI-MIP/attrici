from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from attrici.detrend import Config, detrend


def detrend_run(variable_name):
    config = Config(
        gmt_file=Path("./tests/data/20CRv3-ERA5_germany_ssa_gmt.nc"),
        input_file=Path("./tests/data/20CRv3-ERA5_germany_obs.nc"),
        mask_file=Path("./tests/data/mask_lat50.75_lon9.25.nc"),
        variable=variable_name,
        output_dir=Path("./tests/data/output/scipy"),
        overwrite=True,
        report_variables=["ds", "y", "cfact"],
        solver="scipy",
        stop_date="2021-12-31",
    )
    detrend(config)

    desired = pd.read_hdf(
        f"./tests/data/20CRv3-ERA5_germany_target_{variable_name}_lat50.75_lon9.25.h5"
    )
    actual = pd.read_hdf(
        f"./tests/data/output/scipy/timeseries/{variable_name}/lat_50.75/ts_lat50.75_lon9.25.h5"
    )

    return actual, desired


rtol = 1e-02


@pytest.mark.slow
def test_detrend_run_tas():
    actual, desired = detrend_run("tas")
    np.testing.assert_allclose(actual.cfact, desired.cfact, rtol=rtol, atol=1e-05)
    np.testing.assert_allclose(actual.y, desired.y)


@pytest.mark.slow
def test_detrend_run_tasskew():
    actual, desired = detrend_run("tasskew")
    np.testing.assert_allclose(actual.cfact, desired.cfact, rtol=rtol, atol=1e-05)
    np.testing.assert_allclose(actual.y, desired.y)


@pytest.mark.slow
def test_detrend_run_tasrange():
    actual, desired = detrend_run("tasrange")
    np.testing.assert_allclose(actual.cfact, desired.cfact, rtol=0.19, atol=0.1)
    np.testing.assert_allclose(actual.y, desired.y)


@pytest.mark.slow
def test_detrend_run_pr():
    actual, desired = detrend_run("pr")

    # Days with rain in both
    data = pd.DataFrame({"actual": actual.cfact, "desired": desired.cfact})
    test_data = data[(data.actual > 0) & (data.desired > 0)]
    np.testing.assert_allclose(
        test_data.actual, test_data.desired, rtol=rtol, atol=5e-05
    )

    # Days with rain in one dataset should have no or little rain in the other
    test_data = data[(data.actual <= 0) | (data.desired <= 0)]
    np.testing.assert_allclose(
        test_data.actual, test_data.desired, rtol=rtol, atol=1e-05
    )

    np.testing.assert_allclose(actual.y, desired.y)


@pytest.mark.slow
def test_detrend_run_ps():
    actual, desired = detrend_run("ps")
    np.testing.assert_allclose(actual.cfact, desired.cfact, rtol=rtol, atol=1e-05)
    np.testing.assert_allclose(actual.y, desired.y)


@pytest.mark.slow
def test_detrend_run_hurs():
    actual, desired = detrend_run("hurs")
    np.testing.assert_allclose(actual.cfact, desired.cfact, rtol=0.11, atol=0.1)
    np.testing.assert_allclose(actual.y, desired.y)


@pytest.mark.slow
def test_detrend_run_rsds():
    actual, desired = detrend_run("rsds")
    np.testing.assert_allclose(actual.cfact, desired.cfact, rtol=rtol, atol=1e-05)
    np.testing.assert_allclose(actual.y, desired.y)


@pytest.mark.slow
def test_detrend_run_rlds():
    actual, desired = detrend_run("rlds")
    np.testing.assert_allclose(actual.cfact, desired.cfact, rtol=rtol, atol=1e-05)
    np.testing.assert_allclose(actual.y, desired.y)


@pytest.mark.slow
def test_detrend_run_sfc_wind():
    actual, desired = detrend_run("sfcWind")
    np.testing.assert_allclose(actual.cfact, desired.cfact, rtol=0.16, atol=1e-01)
    np.testing.assert_allclose(actual.y, desired.y)
