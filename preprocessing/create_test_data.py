import sys
import xarray as xr
import os
import numpy as np
from datetime import datetime
if "../" not in sys.path:
    sys.path.append("../")
import settings as s

print('Creating test dataset')
file_to_read = os.path.join(s.data_dir, s.base_file)
print('Inputfile: ' + file_to_read)
file_to_write = os.path.join(
    s.input,
    s.variable
    + '_'
    + s.dataset
    + '_iowa.nc4')
print('Outputfile: ' + file_to_write)
start_time = datetime.now()
data = xr.open_dataset(file_to_read, chunks={'lat': 10, 'lon': 10}).isel(
    #  lat=range(0, 360, 30),
    #  lon=range(0, 720, 30))
    lat=range(2*40-180, 2*44-180),
    lon=range(-2*96+360, -2*92+360))
data.to_netcdf(file_to_write, mode='w')
end_time = datetime.now()
print('Job took ' + str((end_time - start_time).total_seconds()) + 'seconds.')
