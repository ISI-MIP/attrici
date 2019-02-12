#!/home/bschmidt/.programs/anaconda3/envs/detrending/bin/python3

import os
import glob
import xarray as xr
import numpy as np
import dtr_merging
import dtr_smoothing
os.chdir('/p/tmp/bschmidt/')

PATH = '/p/projects/isimip/isimip/ISIMIP2a/InputData/' \
    'climate_co2/climate/HistObs/GSWP3/wind_*'
OUTPUT = '/p/tmp/bschmidt/gswp3/wind_smooth_gswp3.nc4'
data = dtr_merging(PATH)
smooth = dtr_smoothing(data, hws=15)

smooth.to_netcdf(OUTPUT, mode='w')
