import sys
import xarray as xr
import os
import numpy as np
from datetime import datetime
if "../" not in sys.path:
    sys.path.append("../")
import settings as s

sub=5

print('Creating test dataset')
file_to_read = os.path.join(
    s.input_dir,
    s.variable
    + '_'
    + s.dataset
    + '.nc4')
print('Inputfile: ' + file_to_read)
file_to_write = os.path.join(
    s.input_dir,
    s.variable
    + '_'
    + s.dataset
    + '_subset.nc4')
print('Outputfile: ' + file_to_write)
start_time = datetime.now()
data = xr.open_dataset(file_to_read, chunks={'lat': 10, 'lon': 10}).isel(
    lat=range(0, 360, sub),
    lon=range(0, 720, sub))
data.to_netcdf(file_to_write, mode='w')
end_time = datetime.now()
print('Job took ' + str((end_time - start_time).total_seconds()) + 'seconds.')
