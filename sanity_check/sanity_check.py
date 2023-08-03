#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sanity checks for each variable"""


import sys, os
from pathlib import Path
import argparse
import numpy as np
import xarray as xr

import estimation_quality_check as e

#sys.path.insert(0, "../")
import settings as s


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("tile")
    parser.add_argument("variable_hour")
    args = parser.parse_args()

    tile = args.tile 
    variable_hour = args.variable_hour  #e.g tas0 or hurs
    variable = ''.join(i for i in variable_hour if not i.isdigit())
    
    ts_dir = Path(f"/p/tmp/annabu/projects/attrici/output/{tile}/attrici_03_era5_t{tile}_{variable_hour}_rechunked/timeseries/{variable}")
        
    ## check if enough land-cells were processed by comparing number of files with number of land cells
    lsm_file = s.input_dir / f"landmask_{tile}.nc"
    lsm = xr.load_dataset(lsm_file)
    nbr_landcells = lsm["area_European_01min"].count().values.tolist()
    
    nbr_files = e.count_files_in_directory(ts_dir, ".h5")
    print(ts_dir)
    assert  nbr_files == nbr_landcells , f"{nbr_files} number of files <-> {nbr_landcells} number of land cells"
    
    ## check that there are not to many failing cells
    failing_cells = ts_dir.parent.parent / "./failing_cells.log"
    with open(failing_cells, "r") as f:
         nbr_failcells = sum(1 for _ in f)

    assert nbr_failcells == 0 , f"{nbr_failcells} failing cells in tile: {tile}, varibale: {variable_hour}"
    
    
    print("Passed sanity checks")
  
  
if __name__ == "__main__":
    main()
    