#!/usr/bin/env python3

# coding: utf-8

import glob
import itertools as it
import os
import sys
from pathlib import Path

import settings as s
import netCDF4 as nc
import numpy as np


def form_global_nc(ds, time, lat, lon):

    ds.createDimension("time", None)
    ds.createDimension("lat", lat.shape[0])
    ds.createDimension("lon", lon.shape[0])

    times = ds.createVariable("time", "f8", ("time",))
    longitudes = ds.createVariable("lon", "f8", ("lon",))
    latitudes = ds.createVariable("lat", "f8", ("lat",))
    data = ds.createVariable(
        "tas",
        "f4",
        ("time", "lat", "lon"),
        chunksizes=(time.shape[0], 1, 1),
        fill_value=np.nan,
    )
    times.units = "days since 1900-01-01 00:00:00"
    latitudes.units = "degree_north"
    latitudes.long_name = "latitude"
    latitudes.standard_name = "latitude"
    longitudes.units = "degree_east"
    longitudes.long_name = "longitude"
    longitudes.standard_name = "longitude"
    data.missing_value = np.nan
    # FIXME: make flexible or implement loading from source data
    latitudes[:] = lat
    longitudes[:] = lon
    times[:] = time


data_list = glob.glob(str(s.output_dir) + "/timeseries/" + s.variable + "*.nc4")

lat_indices = []
for string in data_list:
    i = string.split("_")[-2]
    lat_indices.append(i)

lon_indices = []
for string in data_list:
    j = string.split("_")[-1].split(".")[0]
    lon_indices.append(j)

print(lon_indices)

obs = nc.Dataset(Path(s.input_dir) / s.source_file, "r")
time = obs.variables["time"][:]
lat = obs.variables["lat"][:]
lon = obs.variables["lon"][:]
cfact = nc.Dataset(Path(s.output_dir) / "cfact" / s.cfact_file, "w", format="NETCDF4")
trend = nc.Dataset(Path(s.output_dir) / "cfact" / s.trend_file, "w", format="NETCDF4")
form_global_nc(cfact, time, lat, lon)
form_global_nc(trend, time, lat, lon)

for (i, j, path) in it.zip_longest(lat_indices, lon_indices, data_list):
    print(i)
    print(j)
    print(path)

    with nc.Dataset(path, "r") as nc_ts:

        obs_ts = obs.variables[s.variable][:, i, j]
        ts = nc_ts.variables[s.variable][:].data
        cfact.variables[s.variable][:, i, j] = ts
        trend.variables[s.variable][:, i, j] = obs_ts - ts
        print(i, j)

print("hello")
obs.close()
cfact.close()
trend.close()
