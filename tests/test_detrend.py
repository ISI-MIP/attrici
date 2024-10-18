import subprocess

import pandas as pd
import pytest


@pytest.mark.slow
def test_detrend_run():
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
        "tas",
        "--stop-date",
        "2023-12-31",
        "--report-variables",
        "ds",
        "y",
        "cfact",
        "logp",
        "--overwrite",
    ]

    status = subprocess.run(command, check=False)
    status.check_returncode()

    reference_data = pd.read_hdf(
        "./tests/data/20CRv3-ERA5_germany_target_tas_lat50.75_lon9.25.h5"
    )
    output = pd.read_hdf(
        "./tests/data/output/timeseries/tas/lat_50.75/ts_lat50.75_lon9.25.h5"
    )

    MAX_DIFFERENCE = 0.4
    assert abs(reference_data.cfact - output.cfact).max() < MAX_DIFFERENCE
