#!/usr/bin/env python3
# coding: utf-8
# """Merge trace and timeseries files into single netcdf files"""

import os, sys
import glob
from pathlib import Path
import argparse
import re
import numpy as np
import xarray as xr
import pickle



def get_float_from_string(file_name):
    """
    Extract floats from foldernames or filenames
    """
    floats_in_string = re.findall(r"[-+]?(?:\d*\.*\d+)", file_name)
    if len(floats_in_string) != 1:
        raise ValueError("there is more than one float in this string")
    return float(floats_in_string[0])
    
    
def merge_files(in_dir):
    """
    Mege trace or timeseries files into single netcdf
    in_dir (str): folder path to file which should be merged to nc
    """
    # Write from single parameter files to netcdf
    parameter_files = []
<<<<<<< HEAD
    for file in in_dir.glob("**/*lon*"):
=======
    for file in in_dir.glob("**/lon*"):
>>>>>>> 047b853ef21c278e66c9b10dd94503a5e0815dca
        lat = get_float_from_string(file.parent.name)
        lon = get_float_from_string(file.stem.split("lon")[-1])
        data_vars = []
        with open(file, "rb") as trace:
            free_params = pickle.load(trace)
        for key in free_params.keys():
            try:
                d = np.arange(len(free_params[key]))
            except TypeError as e:
                if str(e) == "len() of unsized object":
                    d = np.arange(1)
                else:
                    raise e
            data_vars.append(
                xr.DataArray(
                    dims=["lat", "lon", "d"],  #TODO check dim order if same as for final_cfact.nc4
                    data=free_params[key].reshape((1,1,-1)),
                    coords={
                        "lat": ("lat", [lat]),
                        "lon": ("lon", [lon]),
                        "d": ("d", d),
                    },
                    name=key
                )
            )
<<<<<<< HEAD
        parameter_files.append(xr.merge(data_vars))  ## TODO fix appending all cells not only one
=======
        parameter_files.append(xr.merge(data_vars))
>>>>>>> 047b853ef21c278e66c9b10dd94503a5e0815dca
    
    return xr.merge(parameter_files)
    
    
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("tile")   # 5 digits e.g 00001
    parser.add_argument("trace_or_ts") # str "timeseries" or "traces"
    parser.add_argument("variable_hour") # var + hour if exists
    args = parser.parse_args()

    tile = args.tile 
    trace_or_ts = args.trace_or_ts     
    variable_hour = args.variable_hour     
    variable = ''.join(i for i in variable_hour if not i.isdigit())
    
    out_dir = Path(f"/p/tmp/annabu/projects/attrici/output/{tile}/attrici_03_era5_t{tile}_{variable_hour}_rechunked/")
    in_dir = out_dir / trace_or_ts / variable
    
    ## merge files
    merged_parameters = merge_files(in_dir)

    # write to disk
<<<<<<< HEAD
    filepath = in_dir.parent.parent / f"merged_{trace_or_ts}_{tile}_{variable_hour}.nc" 
=======
    filepath = in_dir.parent.parent / f"merged_{trace_or_ts}_{tile}_{variable_hour}.nc4" 
>>>>>>> 047b853ef21c278e66c9b10dd94503a5e0815dca
    merged_parameters.to_netcdf(filepath)

if __name__ == "__main__":
    main()
