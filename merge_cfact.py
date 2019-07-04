#!/usr/bin/env python3

# coding: utf-8

import glob
import itertools as it
import os
import sys
from pathlib import Path

import settings as s
import netCDF4 as nc
import pandas as pd
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


#  data_gen = Path(s.output_dir / "timeseries").glob("**/*.csv")
data_gen = Path("/p/tmp/mengel/isimip/isi-cfact/output/isicf004_gswp3_tas_2p5deg/timeseries").glob("**/*.csv")
data_list = []
lat = []
lon = []
lat_indices = []
lon_indices = []
for i in data_gen:
    data_list.append(str(i))
    lat_float = float(str(i).split("lat")[-1].split("_")[0])
    lon_float = float(str(i).split("lon")[-1].split(".csv")[0])
    lat.append(lat_float)
    lon.append(lon_float)
    lat_indices.append(int(180 - 2*lat_float - .5)/s.subset)
    lon_indices.append(int(2*lon_float - .5 + 360)/s.subset)


obs = nc.Dataset(Path(s.input_dir) / s.source_file, "r")
time = obs.variables["time"][:]
lat = obs.variables["lat"][:]
lon = obs.variables["lon"][:]
headers = pd.read_csv(data_list[0], index_col=0, nrows=1).keys()
headers = headers.drop(["ds", "gmt", "gmt_scaled"])

outs = []
for head in headers:
    outs.append(nc.Dataset(Path(s.output_dir)
                           / "cfact"
                           / (head + "_" + s.cfact_file.split("_", 1)[1]),
                           "w",
                           format="NETCDF4"))
for out in outs:
    form_global_nc(out, time, lat, lon)
#  trend = nc.Dataset(Path(s.output_dir) / "cfact" / s.trend_file, "w", format="NETCDF4")
#  estimate = nc.Dataset(Path(s.output_dir) / "cfact" / s.est_file, "w", format="NETCDF4")
#  form_global_nc(cfact, time, lat, lon)
#  form_global_nc(trend, time, lat, lon)
#  form_global_nc(estimate, time, lat, lon)

for (i, j, path) in it.zip_longest(lat_indices, lon_indices, data_list):
    print(path)
    print(i, j)
    df = pd.read_csv(path, index_col=0, engine='c')
    obs_ts = np.array(obs.variables[s.variable][:, i, j])
    for (out, head) in it.zip_longest(outs, headers):
        ts = np.array(df[head])
        out.variables[s.variable][:, i, j] = ts
    #  ts = np.array(pd.read_csv(path)["cfact"])
    #  cfact.variables[s.variable][:, i, j] = ts
    #  trend.variables[s.variable][:, i, j] = obs_ts - ts
    #  estimate.variables[s.variable][:, i, j] = np.array(pd.read_csv(path)["estimated"])

for out in outs:
    out.close()
#  obs.close()
#  cfact.close()
#  trend.close()
#  estimate.close()
