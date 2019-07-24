#!/usr/bin/env python3

# coding: utf-8

import glob
import itertools as it
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd
from idetrend.datahandler import form_global_nc
import settings as s

#  get input and output
data_gen = Path(s.output_dir / "timeseries").glob("**/*.csv")
out = nc.Dataset(Path(s.output_dir) / "cfact" / s.cfact_file, "w", format="NETCDF4")


data_list = []
lat_indices = []
lon_indices = []
for i in data_gen:
    data_list.append(str(i))
    lat_float = float(str(i).split("lat")[-1].split("_")[0])
    lon_float = float(str(i).split("lon")[-1].split(".csv")[0])
    lat_indices.append(int(180 - 2 * lat_float - 0.5))
    lon_indices.append(int(2 * lon_float - 0.5 + 360))

obs = nc.Dataset(Path(s.input_dir) / s.source_file, "r")
time = obs.variables["time"][:]
lat = obs.variables["lat"][:]
lon = obs.variables["lon"][:]

#  get headers and form empty netCDF file with all meatdata
headers = pd.read_csv(data_list[0], index_col=0, nrows=1).keys()
headers = headers.drop(["y", "y_scaled", "t", "ds", "gmt", "gmt_scaled"])
form_global_nc(out, time, lat, lon, headers, obs.variables["time"].units)

#  adjust indices if datasets are subsets (lat/lon-shapes are smaller than 360/720)
lat_indices *= int(lat.shape[0] / 360)
lon_indices *= int(lon.shape[0] / 720)

for (i, j, path) in it.zip_longest(lat_indices, lon_indices, data_list):
    print(path)
    print("indices are:", i, j)
    df = pd.read_csv(path, index_col=0, engine="c")
    # obs_ts = obs.variables[s.variable][:, i, j]
    # print(np.sum(np.isnan(obs_ts)))
    for head in headers:
        ts = df[head]
        out.variables[head][:, i, j] = np.array(ts)
    print("wrote data from", path, "to", i, j)

out.close()
