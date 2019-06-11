#!/usr/bin/env python3

# coding: utf-8

import os
import sys
import glob
import numpy as np
import netCDF4 as nc
import itertools as it

def form_nc(ds):
    ds.createDimension("time", None)
    ds.createDimension("lat", 360)
    ds.createDimension("lon", 720)

    times = ds.createVariable("time", "f8", ("time",))
    longitudes = ds.createVariable("lon", "f8", ("lon",))
    latitudes = ds.createVariable("lat", "f8", ("lat",))
    data = ds.createVariable("tas", "f4", ("time", "lat", "lon"),
                             chunksizes=(40177, 1, 1),
                             fill_value = np.nan)
    times.units = "days since 1900-01-01 00:00:00"
    latitudes.units = "degree_north"
    latitudes.long_name = "latitude"
    latitudes.standard_name = "latitude"
    longitudes.units = "degree_east"
    longitudes.long_name = "longitude"
    longitudes.standard_name = "longitude"
    data.missing_value = np.nan
    # FIXME: make flexible or implement loading from source data
    latitudes[:] = np.arange(89.75, -89.76, -.5)
    longitudes[:] = np.arange(-179.75, 179.76, .5)
    times[:] = np.arange(40177)

data_list = glob.glob("/home/bschmidt/temp/gswp3/output/detrending/timeseries/tas_gswp3_cfactual_*_*.nc4")

lat_indices = []
for string in data_list:
    i = string.split("_")[-2]
    lat_indices.append(i)

lon_indices = []
for string in data_list:
    j = string.split("_")[-1].split(".")[0]
    lon_indices.append(j)


obs = nc.Dataset("/home/bschmidt/temp/gswp3/input/tas_gswp3_1901_2010.nc4", "r")
cfact = nc.Dataset("/home/bschmidt/temp/gswp3/output/detrending/tas_gswp3_cfactual.nc4", "w", format="NETCDF4")
trend = nc.Dataset("/home/bschmidt/temp/gswp3/output/detrending/tas_gswp3_trend.nc4", "w", format="NETCDF4")
form_nc(cfact)
form_nc(trend)
for (i, j, path) in it.zip_longest(lat_indices, lon_indices, data_list):
    print(i)
    print(j)
    print(path)
    with nc.Dataset(path, "r") as nc_ts:
        obs_ts = obs.variables["tas"][:, i, j]
        ts = nc_ts.variables["tas"][:].data
        cfact.variables["tas"][:, i, j] = ts
        trend.variables["tas"][:, i, j] = obs_ts - ts
        print(i, j)
obs.close()
cfact.close()
trend.close()
