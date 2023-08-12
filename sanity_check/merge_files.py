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
    
    
def merge_files(in_dir, traces_or_ts):
    """
    Mege trace or timeseries files into single netcdf
    in_dir (str): folder path to file which should be merged to nc
    """
    if traces_or_ts=="timeseries":  ## TODO merge ts files in similar way as trace files via xarray
        for file in in_dir.glob("**/*lon*"):
            lat = get_float_from_string(file.parent.name)
            lon = get_float_from_string(file.stem.split("lon")[-1])
        
            hdf = pd.HDFStore(file, mode="r")
            df = hdf.get(hdf.keys()[0])
            ts = {f'{lat}_{lon}': df.to_dict()}
      
            # write single timeseries files to pickle
            tile = re.findall(r"\d{5}", str(in_dir))[0]
            traces_or_timeseries = in_dir.parent.stem
            var_folder = in_dir.parent.parent.name
            filename = "merged_" + traces_or_timeseries + "_" + var_folder[var_folder.find(f't{tile}'):].rsplit("_",1)[0]
            filepath = in_dir.parent.parent / f"{filename}.pickle" 
            try:
                with open(filepath, 'ab+') as f:
                    pickle.dump(ts, f, protocol=pickle.HIGHEST_PROTOCOL)
            except:
                with open(filepath, 'wb') as f:
                    pickle.dump(ts, f, protocol=pickle.HIGHEST_PROTOCOL)

        print("Wrote merged file to:", filepath) 

            ## reload pickle file via:
            # reloaded_merged_ts = []
            # with open(filepath, "rb") as f:
                # #reloaded_merged_ts = pickle.load(f)
                # while True:
                    # try:
                        # reloaded_merged_ts.append(pickle.load(f))
                    # except EOFError:
                        # break

    else:
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
 
            # write to disk
            tile = re.findall(r"\d{5}", str(in_dir)) [0]
            traces_or_timeseries = in_dir.parent.stem
            var_folder = in_dir.parent.parent.name
            filename = "merged_" + traces_or_timeseries + "_" + var_folder[var_folder.find(f't{tile}'):].rsplit("_",1)[0]
            filepath = in_dir.parent.parent / f"{filename}.pickle" 
            try:
                with open(filepath, 'ab+') as f:
                    pickle.dump(xr.merge(data_vars), f, protocol=pickle.HIGHEST_PROTOCOL)
            except:
                with open(filepath, 'wb') as f:
                    pickle.dump(xr.merge(data_vars), f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Wrote merged file to:", filepath) 



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("tile")   # 5 digits e.g 00001
    parser.add_argument("traces_or_ts") # str "timeseries" or "traces"
    parser.add_argument("variable_hour") # var + hour if exists
    args = parser.parse_args()

    tile = args.tile 
    traces_or_ts = args.traces_or_ts     
    variable_hour = args.variable_hour     
    variable = ''.join(i for i in variable_hour if not i.isdigit())
    
    out_dir = Path(f"/p/tmp/annabu/projects/attrici/output/{tile}/attrici_03_era5_t{tile}_{variable_hour}_rechunked/")
    #out_dir = Path(f"/p/tmp/annabu/projects/attrici/{tile}/{tile}/")
    in_dir = out_dir / traces_or_ts / variable
    
    ## merge files and store as nc or pickle
    merge_files(in_dir, traces_or_ts)



if __name__ == "__main__":
    main()
