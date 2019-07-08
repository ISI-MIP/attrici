import sys
import os
import numpy as np
import netCDF4 as nc
from datetime import datetime
if "../" not in sys.path:
    sys.path.append("../")
import settings as s

sub=5

def form_global_nc(ds, time, lat, lon):

    ds.createDimension("time", None)
    ds.createDimension("lat", lat.shape[0])
    ds.createDimension("lon", lon.shape[0])
    print(lat.shape[0])
    print(lon.shape[0])

    times = ds.createVariable("time", "f8", ("time",))
    longitudes = ds.createVariable("lon", "f8", ("lon",))
    latitudes = ds.createVariable("lat", "f8", ("lat",))
    data = ds.createVariable(
        s.variable,
        "f4",
        ("time", "lat", "lon"),
        fill_value=np.nan,
    )
    times.units = "days since 1860-01-01 00:00:00"
    latitudes.units = "degree_north"
    latitudes.long_name = "latitude"
    latitudes.standard_name = "latitude"
    longitudes.units = "degree_east"
    longitudes.long_name = "longitude"
    longitudes.standard_name = "longitude"
    data.missing_value = np.nan
    # FIXME: make flexible or implement loading from source data
    latitudes[:] = lat[:]
    longitudes[:] = lon[:]
    times[:] = time

start_time = datetime.now()
# read metadata
print('Creating test dataset')
file_to_read = os.path.join(
    s.input_dir,
    s.variable
    + '_'
    + s.dataset
    + '.nc4')
print('Inputfile: ' + file_to_read)
read = nc.Dataset(file_to_read, "r")
time = read.variables["time"][:]
lat = read.variables["lat"][::sub].data
lon = read.variables["lon"][::sub].data

#write metadata
file_to_write = os.path.join(
    s.input_dir,
    s.variable
    + '_'
    + s.dataset
    + '_subset.nc4')
print('Outputfile: ' + file_to_write)
write = nc.Dataset(file_to_write, "w", format="NETCDF4")
form_global_nc(write, time, lat, lon)

# actual reading and writing
data = read.variables[s.variable][:, ::sub, ::sub].data
write.variables[s.variable][:] = data

end_time = datetime.now()
print('Job took ' + str((end_time - start_time).total_seconds()) + 'seconds.')
