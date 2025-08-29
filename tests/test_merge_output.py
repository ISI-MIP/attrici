import subprocess
from pathlib import Path

import pytest
from loguru import logger


@pytest.mark.slow
def test_merge_output():
    directory = "./tests/data/output/pymc5/timeseries/tas"

    MIN_COUNT_FILES = 2

    if len(list(Path(directory).glob("**/*.nc"))) < MIN_COUNT_FILES:
        logger.info("Running pymc5 detrend for tas to create files to be merged.")
        solver = "pymc5"
        variable_name = "tas"
        command = [
            "attrici",
            "detrend",
            "--gmt-file",
            "./tests/data/20CRv3-ERA5_germany_ssa_gmt.nc",
            "--input-file",
            "./tests/data/20CRv3-ERA5_germany_obs.nc",
            "--output-dir",
            f"./tests/data/output/{solver}",
            "--variable",
            variable_name,
            "--cells",
            "51.25,9.25;51.25,9.75",
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

    command = [
        "attrici",
        "merge-output",
        directory,
        "./tests/data/output/test-of-merged-tas-output-data.nc",
    ]

    logger.info("Running command: {}", " ".join(command))

    status = subprocess.run(command, check=False)
    status.check_returncode()
