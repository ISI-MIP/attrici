import sys
if "../" not in sys.path: sys.path.append("../")
import os
import numpy as np
import xarray as xr
from datetime import datetime
import settings as s

print('Creating test dataset')
file_to_read = os.path.join(s.data_dir, s.base_file)
print('Inputfile: ' + file_to_read)
file_to_write = os.path.join(s.data_dir, 'test_data_' + s.variable + '.nc4')
print('Outputfile: ' + file_to_write)
start_time = datetime.now()
data = xr.open_dataset(file_to_read, chunks={'lat': 10, 'lon': 10}).isel(
    lat=range(0, 360, 30),
    lon=range(0, 720, 30))
data.to_netcdf(file_to_write, mode='w')
end_time = datetime.now()
print('Job took ' + str((end_time - start_time).total_seconds()) + 'seconds.')
