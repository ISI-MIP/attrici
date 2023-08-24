#!/usr/bin/env python3
# coding: utf-8
# """Merge trace and timeseries files into pickle files"""

import os, sys
import glob
from pathlib import Path
import argparse
import re
import numpy as np
import pandas as pd
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
    
    
def merge_files(in_dir, out_file):
    """
    Mege trace files into single netcdf
    in_dir (str): folder path to file which should be merged to nc
    """
    # Write single trace files to netcdf
    parameter_files = []

    for file in in_dir.glob("**/*lon*"):
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
                    dims=["lat", "lon", "d"],
                    data=free_params[key].reshape((1,1,-1)),
                    coords={
                        "lat": ("lat", [lat]),
                        "lon": ("lon", [lon]),
                        "d": ("d", d),
                    },
                    name=key
                )
            )
        # write to disk
        try:
            with open(out_file, 'ab+') as f:
                pickle.dump(xr.merge(data_vars), f, protocol=pickle.HIGHEST_PROTOCOL)
        except:
            with open(out_file, 'wb') as f:
                pickle.dump(xr.merge(data_vars), f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Wrote merged traces to: ", out_file) 



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("tile")   # 5 digits e.g 00001
    parser.add_argument("variable_hour") # var + hour if exists
    args = parser.parse_args()

    tile = args.tile    
    variable_hour = args.variable_hour     
    variable = ''.join(i for i in variable_hour if not i.isdigit())

    
    in_dir = Path(f"/p/projects/ou/rd3/dmcci/basd_era5-land_to_efas-meteo/attrici_output_anna/storage/{tile}/attrici_03_era5_t{tile}_{variable_hour}_rechunked/traces/{variable}")  # if files stored in project folder
    ## merge trace files and store as single pickle

    tile = re.findall(r"\d{5}", str(in_dir)) [0]
    var_folder = in_dir.parent.parent.name     
    filename = "merged_traces"  + "_" + var_folder[var_folder.find(f't{tile}'):].rsplit("_",1)[0]
    out_file = in_dir.parent.parent / f"{filename}.pickle" 

    merge_files(in_dir, out_file)
    
    if not out_file.is_file(): # if files stored in tmp folder
        in_dir = Path(f"/p/tmp/annabu/projects/attrici/output/{tile}/attrici_03_era5_t{tile}_{variable_hour}_rechunked/traces/{variable}")    
        out_file = in_dir.parent.parent / f"{filename}.pickle" 
        merge_files(in_dir, out_file)   



if __name__ == "__main__":
    main()
