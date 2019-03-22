import os
import numpy as np
import xarray as xr
from datetime import datetime
import settings as s

file_to_read = os.path.join(s.data_dir, s.base_file)
file_to_write = os.path.join(s.data_dir, s.to_detrend_file)
start_time = datetime.now()
data = xr.open_dataset(file_to_read, chunks={'lat': 10, 'lon': 10}).isel(
    lat=range(0, 360, 30),
    lon=range(0, 720, 30))
data.to_netcdf(file_to_write, mode='w')
