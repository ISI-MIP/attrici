import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import os
import sys
from scipy import stats
import imp
import time as t

# for Ben and me to use our code.
if "../" not in sys.path: sys.path.append("../")
import idetrend.visualization as vis; imp.reload(vis)
import settings as s

# parameters and data paths
source_path = "/p/tmp/bschmidt/gswp3/"

# the file with the smoothed global trend of global mean temperature
gmt_data_path = os.path.join(source_path, 'test_ssa_gmt.nc4')
# the daily interpolated ssa-smoothed global mean temperature
gmt_on_each_day = vis.get_gmt_on_each_day(gmt_data_path)

varname = s.variable
file_to_write = os.path.join(source_path, varname+'_detrended.nc4')

# the full dataset
regr_path = os.path.join(source_path, varname+'_regression_all.nc4')
data_path = os.path.join(source_path,varname+'_rm_gswp3_1901_2010.nc4')

#test data set
#regr_path = os.path.join(source_path, varname+'_regression.nc4')
#data_path = os.path.join(source_path,'test_data_'+varname+'.nc4')

lats, lons, slope, intercept = vis.get_coefficient_fields(regr_path)

if s.test:
    shape_of_input = (40150, 12, 24)
else:
    shape_of_input = (40150, 360, 720)

# functions
def fit_ts(regr_path, data_path, variable, gmt_on_each_day, indices, shape_of_input, calendar='noleap'):

    """ detrend 2-dimensional data with linear trend from regression coefficients. """

    if calendar == "noleap":
        days_of_year = 365

    lat, lon, slope, intercept = vis.get_regression_coefficients(regr_path, indices)
    data_to_detrend = vis.get_data_to_detrend(data_path, variable, indices)
    gmt_on_doy = np.ones((110, shape_of_input[1], shape_of_input[2]))
    for lat in range(shape_of_input[1]):
        for lon in range(shape_of_input[2]):
            gmt_on_doy[:, lat, lon] = gmt_on_each_day[indices[0]::days_of_year]

    fit = intercept + (slope * gmt_on_doy)
    data_detrended = data_to_detrend - fit + fit[0, :, :]

    return data_detrended

def write_detrended(regr_path, data_path, shape_of_input, original_data_coords, file_to_write, variable):

    """ datrend data and write it to netCDF file. """

    # create data set and dimensions
    output_ds = nc.Dataset(file_to_write, "w", format="NETCDF4")

    time = output_ds.createDimension("time", None)
    lat = output_ds.createDimension("lat", original_data_coords[0].shape[0])
    lon = output_ds.createDimension("lon", original_data_coords[1].shape[0])
    # print(output_ds.dimensions)

    # create variables
    times = output_ds.createVariable("time", "f8", ("time",))
    longitudes = output_ds.createVariable("lon", "f4", ("lon",))
    latitudes = output_ds.createVariable("lat", "f4", ("lat",))
    data = output_ds.createVariable(variable, "f8", ("time", "lat", "lon",))
    # print(intercepts)

    # Set attributes
    output_ds.description = "Detrended data of variable " + variable
    output_ds.history = "Created " + t.ctime(t.time())
    latitudes.units = "degrees_north"
    latitudes.long_name = "latitudes"
    longitudes.units = "degrees_east"
    longitudes.long_name = "longitudes"
    data.units = "K"
    times.units = "days since 1901-01-01 00:00:00.0"
    times.calendar = "noleap"

    if times.calendar == "noleap":
        days_of_year = 365
        doys = range(1, 366)

    lats = original_data_coords[0][:]
    lons = original_data_coords[1][:]

    latitudes[:] = lats
    longitudes[:] = lons
    times[:] = range(shape_of_input[0])

    #  latis = np.arange(shape_of_input[1])
    #  lonis = np.arange(shape_of_input[2])

    for doy in doys:
        print('Working on doy: ' + str(doy))
        data_detrended = fit_ts(regr_path, data_path, varname, gmt_on_each_day, [doy-1], shape_of_input)
        data[doy-1::days_of_year, :, :] = data_detrended
    output_ds.close()

if __name__ == "__main__":

    TIME0 = datetime.now()

    write_detrended(regr_path, data_path, shape_of_input, (lats, lons), file_to_write, varname)

    TIME1 = datetime.now()
    duration = TIME1 - TIME0
    print('Calculation took', duration.total_seconds(), 'seconds.')
