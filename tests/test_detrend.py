import subprocess

import numpy as np
import pandas as pd
import pytest
from loguru import logger


def detrend_run(variable_name):
    command = [
        "attrici",
        "detrend",
        "--gmt-file",
        "./tests/data/20CRv3-ERA5_germany_ssa_gmt.nc",
        "--input-file",
        "./tests/data/20CRv3-ERA5_germany_obs.nc",
        "--mask-file",
        "./tests/data/mask_lat50.75_lon9.25.nc",
        "--output-dir",
        "./tests/data/output",
        "--variable",
        variable_name,
        "--stop-date",
        "2021-12-31",
        "--report-variables",
        "ds",
        "y",
        "cfact",
        "logp",
        "--overwrite",
    ]

    logger.info("Running command: {}", " ".join(command))

    status = subprocess.run(command, check=False)
    status.check_returncode()

    desired = pd.read_hdf(
        f"./tests/data/20CRv3-ERA5_germany_target_{variable_name}_lat50.75_lon9.25.h5"
    )
    actual = pd.read_hdf(
        f"./tests/data/output/timeseries/{variable_name}/lat_50.75/ts_lat50.75_lon9.25.h5"
    )

    return actual, desired


@pytest.mark.slow
def test_detrend_run_tas():
    actual, desired = detrend_run("tas")
    np.testing.assert_allclose(actual.cfact, desired.cfact)
    np.testing.assert_allclose(actual.y, desired.y)
    np.testing.assert_allclose(actual.logp, desired.logp)


@pytest.mark.slow
def test_detrend_run_tasskew():
    actual, desired = detrend_run("tasskew")
    np.testing.assert_allclose(actual.cfact, desired.cfact)
    np.testing.assert_allclose(actual.y, desired.y)
    np.testing.assert_allclose(actual.logp, desired.logp)


@pytest.mark.slow
def test_detrend_run_tasrange():
    # Unit of tasrange: K
    actual, desired = detrend_run("tasrange")
    np.testing.assert_allclose(actual.cfact, desired.cfact, rtol=1e-06, atol=1e-05)
    np.testing.assert_allclose(actual.y, desired.y)

    # Skipping logp comparison for variables with inconsistent prior distributions
    # Fixed in https://github.com/ISI-MIP/attrici/pull/101
    # np.testing.assert_allclose(actual.logp, desired.logp)


@pytest.mark.slow
def test_detrend_run_pr():
    # Unit of pr: kg m-2 s-1
    actual, desired = detrend_run("pr")

    # Days with rain in both
    data = pd.DataFrame({"actual": actual.cfact, "desired": desired.cfact})
    test_data = data[(data.actual > 0) & (data.desired > 0)]
    np.testing.assert_allclose(test_data.actual, test_data.desired, atol=1e-07)

    # Days with rain in one dataset should have no or little rain in the other
    test_data = data[(data.actual <= 0) | (data.desired <= 0)]
    np.testing.assert_allclose(test_data.actual, test_data.desired, atol=1e-05)

    np.testing.assert_allclose(actual.y, desired.y)

    # Skipping logp comparison for variables with inconsistent prior distributions
    # Fixed in https://github.com/ISI-MIP/attrici/pull/101
    # np.testing.assert_allclose(actual.logp, desired.logp)


@pytest.mark.slow
def test_detrend_run_ps():
    actual, desired = detrend_run("ps")
    np.testing.assert_allclose(actual.cfact, desired.cfact)
    np.testing.assert_allclose(actual.y, desired.y)
    np.testing.assert_allclose(actual.logp, desired.logp)


@pytest.mark.slow
def test_detrend_run_hurs():
    actual, desired = detrend_run("hurs")
    np.testing.assert_allclose(actual.cfact, desired.cfact)
    np.testing.assert_allclose(actual.y, desired.y)
    np.testing.assert_allclose(actual.logp, desired.logp)


@pytest.mark.slow
def test_detrend_run_rsds():
    actual, desired = detrend_run("rsds")
    np.testing.assert_allclose(actual.cfact, desired.cfact)
    np.testing.assert_allclose(actual.y, desired.y)
    np.testing.assert_allclose(actual.logp, desired.logp)


@pytest.mark.slow
def test_detrend_run_rlds():
    actual, desired = detrend_run("rlds")
    np.testing.assert_allclose(actual.cfact, desired.cfact)
    np.testing.assert_allclose(actual.y, desired.y)
    np.testing.assert_allclose(actual.logp, desired.logp)


@pytest.mark.slow
def test_detrend_run_sfc_wind():
    actual, desired = detrend_run("sfcWind")
    np.testing.assert_allclose(actual.cfact, desired.cfact)
    np.testing.assert_allclose(actual.y, desired.y)
    np.testing.assert_allclose(actual.logp, desired.logp)
