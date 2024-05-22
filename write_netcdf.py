#!/usr/bin/env python3
# coding: utf-8

import glob
import itertools
import re
import subprocess
# import pandas as pd
from datetime import datetime
from pathlib import Path

import attrici
import attrici.postprocess as pp
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import xarray as xr

import settings as s

### options for postprocess
write_netcdf = True
rechunk = True
replace_invalid = True
# cdo_processing needs rechunk
cdo_processing = False


# append later with more variables if needed
vardict = {
    s.variable: "cfact",
    s.variable + "_orig": "y",
    # "mu":"mu",
    # "y_scaled": "y_scaled",
    # "pbern": "pbern",
    "logp": "logp",
}

cdo_ops = {
    # "monmean": "monmean ",
    "yearmean": "yearmean ",
    #    "monmean_valid": "monmean -setrtomiss,-1e20,1.1574e-06 -selvar,cfact,y",
    # "yearmean_valid": "yearmean -setrtomiss,-1e20,1.1574e-06 -selvar,cfact,y",
    "trend": "trend ",
    # "mse": "timmean -expr,'squared_error=sqr(mu-y_scaled)'"
    # "trend_valid": "trend -setrtomiss,-1e20,1.1574e-06 -selvar,cfact,y",
}


def get_float_from_string(file_name):
    floats_in_string = re.findall(r"[-+]?(?:\d*\.*\d+)", file_name)
    if len(floats_in_string) != 1:
        raise ValueError("there is no ore more than one float in this string")
    return float(floats_in_string[0])


TIME0 = datetime.now()

source_file = Path(s.input_dir) / s.source_file
ts_dir = s.output_dir / "timeseries" / s.variable
cfact_dir = s.output_dir / "cfact" / s.variable
cfact_file = cfact_dir / s.cfact_file
cfact_rechunked = str(cfact_file).rstrip(".nc4") + "_rechunked.nc4"


if write_netcdf:
    data_gen = ts_dir.glob("**/*" + s.storage_format)
    cfact_dir.mkdir(parents=True, exist_ok=True)

    ### check which data is available
    data_list = []
    # lat_indices = []
    # lon_indices = []
    lat_float_list = []
    lon_float_list = []
    for i in data_gen:
        data_list.append(str(i))
        lat_float = float(str(i).split("lat")[-1].split("_")[0])
        lon_float = float(str(i).split("lon")[-1].split(s.storage_format)[0])
        lat_float_list.append(lat_float)
        lon_float_list.append(lon_float)

    # adjust indices if datasets are subsets (lat/lon-shapes are smaller than 360/720)
    # TODO: make this more robust
    # lat_indices = np.array(np.array(lat_indices) / s.lateral_sub, dtype=int)
    # lon_indices = np.array(np.array(lon_indices) / s.lateral_sub, dtype=int)

    #  get headers and form empty netCDF file with all meatdata
    print(data_list[0])

    # write empty outfile to netcdf with all orignal attributes
    source_data = xr.open_dataset(source_file)
    attributes = source_data[s.variable].attrs
    coords = source_data[s.variable].coords

    outfile = source_data.drop(s.variable)

    outfile.to_netcdf(cfact_file)

    # open with netCDF4 for memory efficient writing
    outfile = nc.Dataset(cfact_file, "a")

    for var in s.report_to_netcdf:
        ncvar = outfile.createVariable(
            var,
            "f4",
            ("time", "lat", "lon"),
            chunksizes=(len(coords["time"]), 1, 1),
            fill_value=9.9692e36,
        )
        if var in [s.variable, s.variable + "_orig"]:
            for key, att in attributes.items():
                ncvar.setncattr(key, att)

    outfile.setncattr("cfact_version", attrici.__version__)
    outfile.setncattr("runid", Path.cwd().name)

    n_written_cells = 0
    for dfpath in data_list:
        try:
            df = pp.read_from_disk(dfpath)
        except ValueError as e:
            print(f"A ValueError was raised when trying to read data from {dfpath}")
            raise e
        lat = get_float_from_string(Path(dfpath).parent.name)
        lon = get_float_from_string(Path(dfpath).stem.split("lon")[-1])

        lat_idx = (np.abs(outfile.variables["lat"][:] - lat)).argmin()
        lon_idx = (np.abs(outfile.variables["lon"][:] - lon)).argmin()

        for var in s.report_to_netcdf:
            ts = df[vardict[var]]
            outfile.variables[var][:, lat_idx, lon_idx] = np.array(ts)
        n_written_cells = n_written_cells + 1

    outfile.close()

    print(
        f"Successfully wrote data from {n_written_cells} cells ",
        f"to the {cfact_file} file.",
    )
    print(
        "Writing took {0:.1f} minutes.".format(
            (datetime.now() - TIME0).total_seconds() / 60
        )
    )

if rechunk:
    cfact_rechunked = pp.rechunk_netcdf(cfact_file, cfact_rechunked)

if replace_invalid:
    cfact_rechunked = pp.replace_nan_inf_with_orig(
        s.variable, source_file, cfact_rechunked
    )


if cdo_processing:
    for cdo_op in cdo_ops:
        outfile = str(cfact_file).rstrip(".nc4") + "_" + cdo_op + ".nc4"
        if "trend" in cdo_op:
            outfile = (
                outfile.rstrip(".nc4") + "_1.nc4 " + outfile.rstrip(".nc4") + "_2.nc4"
            )
        try:
            cmd = "cdo " + cdo_ops[cdo_op] + " " + cfact_rechunked + " " + outfile
            print(cmd)
            subprocess.check_call(cmd, shell=True)
        except subprocess.CalledProcessError:
            cmd = "module load cdo && " + cmd
            print(cmd)
            subprocess.check_call(cmd, shell=True)
