#!/usr/bin/env python
# coding: utf-8
# Sanity checks for each variable


import sys, os
from pathlib import Path
import argparse
import numpy as np
import xarray as xr

import estimation_quality_check as e
import count_replaced_values as c
import settings as s


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("tile")
    parser.add_argument("variable_hour")
    args = parser.parse_args()

    tile = args.tile 
    variable_hour = args.variable_hour  #e.g tas0 or hurs
    variable = ''.join(i for i in variable_hour if not i.isdigit())
 
 
    ## check if enough land-cells were processed by comparing number of files with number of land cells
    lsm_file = s.input_dir / f"landmask_{tile}.nc"
    lsm = xr.load_dataset(lsm_file)
    nbr_landcells = lsm["area_European_01min"].count().values.tolist()
    print(f"{tile}, {variable_hour}: {nbr_landcells} Land cells in lsm" )


    ts_dir = Path(f"/p/tmp/dominikp/attrici/{tile}/attrici_03_era5_t{tile}_{variable_hour}_rechunked/timeseries/")
    print("Searching in",ts_dir)
    nbr_files = e.count_files_in_directory(ts_dir, ".h5")
      
    if nbr_files == 0: # if files stored in tmp folder
        ts_dir = Path(f"/p/tmp/annabu/projects/attrici/output/{tile}/attrici_03_era5_t{tile}_{variable_hour}_rechunked/timeseries/")
        print("Searching in",ts_dir)
        nbr_files = e.count_files_in_directory(ts_dir, ".h5")
  
    assert  nbr_files == nbr_landcells , f"{nbr_files} number of timeseries files <-> {nbr_landcells} number of land cells"   
    
    
    ## ckeck for empty trace or timeseries file, due that some folders were moved by "rsync" but with --partial flag
    ts_files = ts_dir.rglob(f"*.h5")
    assert all([os.stat(file).st_size != 0  for file in ts_files]),  f"empty files exists in {ts_dir}"
    
    trace_dir = ts_dir.parent / "traces"
    trace_files = trace_dir.rglob(f"lon*")
    assert all([os.stat(file).st_size != 0  for file in trace_files]),  f"empty files exists in {trace_dir}"   
    
    
    ## check amount of failing cells
    failing_cells = ts_dir.parent / "./failing_cells.log"
    with open(failing_cells, "r") as f:
         nbr_failcells = sum(1 for _ in f)

    assert nbr_failcells == 0 , f"failing cells in tile: {tile}, variable: {variable_hour}"
        
  
    print("Passed all sanity checks")
  
  
if __name__ == "__main__":
    main()
    
