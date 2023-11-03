#!/usr/bin/env python
# coding: utf-8
# Sanity checks for each variable


import argparse
import os
import sys
from pathlib import Path

import numpy as np
import xarray as xr

import settings as s


def count_files_in_directory(directory, file_extension):
    path = Path(directory)
    files = path.rglob(f"*{file_extension}")
    count = sum(1 for file in files if file.is_file())
    return count


def main():
    # check if enough land-cells were processed
    # by comparing number of files with number of land cells
    lsm_file = s.input_dir / s.landsea_file
    lsm = xr.load_dataset(lsm_file)
    nbr_landcells = lsm["area_European_01min"].count().values.tolist()
    print(f"{s.tile}, {s.variable}{s.hour}: {nbr_landcells} land cells in lsm")

    ts_dir = s.output_dir / "timeseries"
    print("Searching in", ts_dir)
    nbr_files = count_files_in_directory(ts_dir, ".h5")

    assert nbr_files == nbr_landcells, (
        f"{nbr_files} number of timeseries files <-> {nbr_landcells} "
        + "number of land cells"
    )

    # check for empty trace or timeseries file
    ts_files = ts_dir.rglob("*.h5")
    assert all(
        os.stat(file).st_size != 0 for file in ts_files
    ), f"empty files exists in {ts_dir}"

    trace_dir = s.output_dir / "traces"
    trace_files = trace_dir.rglob("lon*")
    assert all(
        os.stat(file).st_size != 0 for file in trace_files
    ), f"empty files exists in {trace_dir}"

    # check amount of failing cells
    failing_cells = ts_dir.parent / "./failing_cells.log"
    with open(failing_cells, "r") as file:
        nbr_failcells = sum(1 for _ in file)

    assert (
        nbr_failcells == 0
    ), f"failing cells in tile: {s.tile}, variable: {s.variable}{s.hour}"

    print("Passed all sanity checks")


if __name__ == "__main__":
    main()
