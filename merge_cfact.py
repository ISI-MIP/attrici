#!/usr/bin/env python3

# coding: utf-8

import glob
import itertools as it
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd
from datetime import datetime
import idetrend.datahandler as dh
import settings as s


TIME0 = datetime.now()

ts_dir = s.output_dir / "timeseries" / s.variable
cfact_dir = s.output_dir / "cfact" / s.variable

data_gen = ts_dir.glob("**/*" + s.storage_format)
cfact_dir.mkdir(parents=True, exist_ok=True)
cfact_file = cfact_dir / s.cfact_file
# cfact_file.unlink()
out = nc.Dataset(cfact_file, "w", format="NETCDF4")

data_list = []
lat_indices = []
lon_indices = []
for i in data_gen:
    data_list.append(str(i))
    lat_float = float(str(i).split("lat")[-1].split("_")[0])
    lon_float = float(str(i).split("lon")[-1].split(s.storage_format)[0])
    lat_indices.append(int(180 - 2 * lat_float - 0.5))
    lon_indices.append(int(2 * lon_float - 0.5 + 360))

obs = nc.Dataset(Path(s.input_dir) / s.source_file, "r")
time = obs.variables["time"][:]
lat = obs.variables["lat"][:]
lon = obs.variables["lon"][:]


#  get headers and form empty netCDF file with all meatdata
headers = dh.read_from_disk(data_list[0]).keys()
headers = headers.drop(["y", "t", "ds", "gmt", "gmt_scaled"])
dh.form_global_nc(out, time, lat, lon, headers, obs.variables["time"].units)

# adjust indices if datasets are subsets (lat/lon-shapes are smaller than 360/720)
# TODO: make this more robust
lat_indices = np.array(lat_indices) / s.lateral_sub
lon_indices = np.array(lon_indices) / s.lateral_sub

for (i, j, dfpath) in it.zip_longest(lat_indices, lon_indices, data_list):

    df = dh.read_from_disk(dfpath)
    for head in headers:
        ts = df[head]
        out.variables[head][:, i, j] = np.array(ts)
    print("wrote data from", dfpath, "to", i, j)

out.close()
print("Successfully wrote", cfact_file, "file. Took")
print("It took {0:.1f} minutes.".format((datetime.now() - TIME0).total_seconds() / 60))
