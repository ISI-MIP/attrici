import subprocess

import pandas as pd
import pytest
from loguru import logger


def detrend_run(variable_name, max_difference):
    command = [
        "attrici",
        "detrend",
        "--gmt-file",
        "./tests/data/20CRv3-ERA5_germany_ssa_gmt.nc",
        "--input-file",
        "./tests/data/20CRv3-ERA5_germany_obs.nc",
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

    reference_data = pd.read_hdf(
        f"./tests/data/20CRv3-ERA5_germany_target_{variable_name}_lat50.75_lon9.25.h5"
    )
    output = pd.read_hdf(
        f"./tests/data/output/timeseries/{variable_name}/lat_50.75/ts_lat50.75_lon9.25.h5"
    )

    assert abs(reference_data.cfact - output.cfact).max() < max_difference


@pytest.mark.slow
def test_detrend_run_tas():
    detrend_run("tas", 1e-9)


@pytest.mark.slow
def test_detrend_run_tasskew():
    detrend_run("tasskew", 1e-9)


@pytest.mark.slow
def test_detrend_run_tasrange():
    detrend_run("tasrange", 1e-9)


@pytest.mark.slow
def test_detrend_run_pr():
    detrend_run("pr", 1e-5)


@pytest.mark.slow
def test_detrend_run_ps():
    detrend_run("ps", 1e-7)


@pytest.mark.slow
def test_detrend_run_hurs():
    detrend_run("hurs", 1e-5)


@pytest.mark.slow
def test_detrend_run_rsds():
    detrend_run("rsds", 1e-7)


@pytest.mark.slow
def test_detrend_run_rlds():
    detrend_run("rlds", 1e-9)


@pytest.mark.slow
def test_detrend_run_sfc_wind():
    detrend_run("sfcWind", 1e-11)
