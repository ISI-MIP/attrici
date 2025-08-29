import netCDF4  # noqa Used to suppress warning https://github.com/pydata/xarray/issues/7259
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import subprocess
import importlib
from loguru import logger

solvers = []
if importlib.util.find_spec("pymc3") is not None:
    solvers.append("pymc3")
if importlib.util.find_spec("pymc") is not None:
    solvers.append("pymc5")
if len(solvers) == 0:
    raise ImportError("Neither PyMC3 nor PyMC5 could be imported")


def detrend_run(variable_name, solver):
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
        f"./tests/data/output/{solver}",
        "--variable",
        variable_name,
        "--stop-date",
        "2021-12-31",
        "--report-variables",
        "y",
        "cfact",
        "logp",
        "--solver",
        solver,
        "--overwrite",
    ]

    logger.info("Running command: {}", " ".join(command))

    status = subprocess.run(command, check=False)
    status.check_returncode()

    desired = pd.read_hdf(
        f"./tests/data/20CRv3-ERA5_germany_target_{variable_name}_lat50.75_lon9.25.h5"
    )
    desired = desired.set_index(desired.ds)
    desired.index.name = "time"
    desired = desired.drop("ds", axis=1)

    actual = xr.load_dataset(
        f"./tests/data/output/{solver}/timeseries/{variable_name}/lat_50.75/ts_lat50.75_lon9.25.nc"
    ).to_dataframe()

    return actual, desired


@pytest.mark.parametrize("solver", solvers)
@pytest.mark.slow
def test_detrend_run_tas(solver):
    actual, desired = detrend_run("tas", solver)
    np.testing.assert_allclose(actual.cfact, desired.cfact)
    np.testing.assert_allclose(actual.y, desired.y)
    np.testing.assert_allclose(actual.logp, desired.logp)


@pytest.mark.parametrize("solver", solvers)
@pytest.mark.slow
def test_detrend_run_tasskew(solver):
    actual, desired = detrend_run("tasskew", solver)
    np.testing.assert_allclose(actual.cfact, desired.cfact)
    np.testing.assert_allclose(actual.y, desired.y)
    np.testing.assert_allclose(actual.logp, desired.logp)


@pytest.mark.parametrize("solver", solvers)
@pytest.mark.slow
def test_detrend_run_tasrange(solver):
    # Unit of tasrange: K
    actual, desired = detrend_run("tasrange", solver)
    np.testing.assert_allclose(actual.cfact, desired.cfact, rtol=1e-06, atol=1e-05)
    np.testing.assert_allclose(actual.y, desired.y)

    # Skipping logp comparison for variables with inconsistent,
    # prior distributions, fixed in https://github.com/ISI-MIP/attrici/pull/101
    # np.testing.assert_allclose(actual.logp, desired.logp)


@pytest.mark.parametrize("solver", solvers)
@pytest.mark.slow
def test_detrend_run_pr(solver):
    # Unit of pr: kg m-2 s-1
    actual, desired = detrend_run("pr", solver)

    # Days with rain in both
    data = pd.DataFrame(
        {"actual": actual.cfact.values, "desired": desired.cfact.values}
    )
    test_data = data[(data.actual > 0) & (data.desired > 0)]
    np.testing.assert_allclose(test_data.actual, test_data.desired, atol=1e-07)

    # Days with rain in one dataset should have no or little rain in the other
    test_data = data[(data.actual <= 0) | (data.desired <= 0)]
    np.testing.assert_allclose(test_data.actual, test_data.desired, atol=1e-05)

    np.testing.assert_allclose(actual.y, desired.y)

    # Skipping logp comparison for variables with inconsistent
    # prior distributions, fixed in https://github.com/ISI-MIP/attrici/pull/101
    # np.testing.assert_allclose(actual.logp, desired.logp)


@pytest.mark.parametrize("solver", solvers)
@pytest.mark.slow
def test_detrend_run_ps(solver):
    actual, desired = detrend_run("ps", solver)
    np.testing.assert_allclose(actual.cfact, desired.cfact)
    np.testing.assert_allclose(actual.y, desired.y)
    np.testing.assert_allclose(actual.logp, desired.logp)


@pytest.mark.parametrize("solver", solvers)
@pytest.mark.slow
def test_detrend_run_hurs(solver):
    actual, desired = detrend_run("hurs", solver)
    np.testing.assert_allclose(actual.cfact, desired.cfact)
    np.testing.assert_allclose(actual.y, desired.y)
    np.testing.assert_allclose(actual.logp, desired.logp)


@pytest.mark.parametrize("solver", solvers)
@pytest.mark.slow
def test_detrend_run_rsds(solver):
    actual, desired = detrend_run("rsds", solver)
    np.testing.assert_allclose(actual.cfact, desired.cfact)
    np.testing.assert_allclose(actual.y, desired.y)
    np.testing.assert_allclose(actual.logp, desired.logp)


@pytest.mark.parametrize("solver", solvers)
@pytest.mark.slow
def test_detrend_run_rlds(solver):
    actual, desired = detrend_run("rlds", solver)
    np.testing.assert_allclose(actual.cfact, desired.cfact)
    np.testing.assert_allclose(actual.y, desired.y)
    np.testing.assert_allclose(actual.logp, desired.logp)


@pytest.mark.parametrize("solver", solvers)
@pytest.mark.slow
def test_detrend_run_sfc_wind(solver):
    actual, desired = detrend_run("sfcWind", solver)
    np.testing.assert_allclose(actual.cfact, desired.cfact)
    np.testing.assert_allclose(actual.y, desired.y)
    np.testing.assert_allclose(actual.logp, desired.logp)
