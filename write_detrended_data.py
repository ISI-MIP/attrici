import os
import sys
import importlib
import time as t
from datetime import datetime
import numpy as np
import netCDF4 as nc
import idetrend as idtr
import idetrend.visualization as vis

importlib.reload(vis)
import settings as s

# parameters and data paths

# the file with the smoothed global trend of global mean temperature
gmt_file = os.path.join(s.data_dir, s.gmt_file)
# the daily interpolated ssa-smoothed global mean temperature
# gmt_on_each_day = vis.get_gmt_on_each_day(gmt_file)
gmt_on_each_day = idtr.utility.get_gmt_on_each_day(gmt_file, s.days_of_year)


varname = s.variable
if s.test:
    file_to_write = os.path.join(s.data_dir, varname + "_detrended_test.nc4")
else:
    file_to_write = os.path.join(s.data_dir, varname + "_detrended.nc4")

regression_file = os.path.join(s.data_dir, s.regression_outfile)
to_detrend_file = os.path.join(s.data_dir, s.to_detrend_file)


lats, lons, slope, intercept = vis.get_coefficient_fields(regression_file)

if s.test:
    shape_of_input = (40150, 12, 24)
else:
    shape_of_input = (40150, 360, 720)

# functions
def fit_minimal(gmt_on_doy, intercept, slope, transform):

    """ A minimal fit function for 1-dimensional data with trend from regression coefficients"""
    if transform is None:
        fit = intercept + (slope * gmt_on_doy)
    else:
        fit = transform[1](intercept + (slope * gmt_on_doy))

    return fit


def fit_ts(
    regr_path,
    data_path,
    variable,
    gmt_on_each_day,
    indices,
    shape_of_input,
    transform=None,
    calendar="noleap",
):

    """ A function to fit 2-dimensional data with trend from regression coefficients.
    Employs fit_minimal()."""

    if calendar == "noleap":
        days_of_year = 365

    lat, lon, slope, intercept = vis.get_regression_coefficients(regr_path, indices, slope="max")
    data_to_detrend = vis.get_data_to_detrend(data_path, variable, indices)
    gmt_on_doy = np.ones((110, shape_of_input[1], shape_of_input[2]))
    for lat in range(shape_of_input[1]):
        for lon in range(shape_of_input[2]):
            gmt_on_doy[:, lat, lon] = gmt_on_each_day[indices[0] :: days_of_year]

    fit = fit_minimal(gmt_on_doy, intercept, slope, transform)
    data_detrended = data_to_detrend - fit + fit[0, :, :]

    return data_detrended


def write_detrended(
    regr_path, data_path, shape_of_input, original_data_coords, file_to_write, variable
):

    """ datrend data and write it to netCDF file. """

    if os.path.exists(file_to_write):
        os.remove(file_to_write)

    # create data set and dimensions
    output_ds = nc.Dataset(file_to_write, "w", format="NETCDF4")

    time = output_ds.createDimension("time", None)
    lat = output_ds.createDimension("lat", original_data_coords[0].shape[0])
    lon = output_ds.createDimension("lon", original_data_coords[1].shape[0])
    # print(output_ds.dimensions)

    # create variables
    times = output_ds.createVariable("time", "f8", ("time",))
    longitudes = output_ds.createVariable("lon", "f8", ("lon",))
    latitudes = output_ds.createVariable("lat", "f8", ("lat",))
    data = output_ds.createVariable(variable, "f4", ("time", "lat", "lon"))
    data_max = output_ds.createVariable(variable + "_max", "f4", ("time", "lat", "lon"))
    data_min = output_ds.createVariable(variable + "_min", "f4", ("time", "lat", "lon"))
    # print(intercepts)

    # Set attributes
    output_ds.description = "Detrended data of variable " + variable
    output_ds.history = "Created " + t.ctime(t.time())
    latitudes.units = "degrees_north"
    latitudes.long_name = "latitude"
    longitudes.standard_name = "latitude"
    longitudes.units = "degrees_east"
    longitudes.long_name = "longitude"
    longitudes.standard_name = "longitude"
    data.units = ""
    times.units = "days since 1901-01-01"
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
        print("Working on doy: " + str(doy))
        # Detrending with average parameters
        data_detrended = fit_ts(
            regr_path, data_path, varname, gmt_on_each_day, [doy - 1], shape_of_input, transform=s.transform[s.variable]
        )
        data[doy - 1 :: days_of_year, :, :] = data_detrended
    output_ds.close()


if __name__ == "__main__":

    TIME0 = datetime.now()

    write_detrended(
        regression_file,
        to_detrend_file,
        shape_of_input,
        (lats, lons),
        file_to_write,
        varname,
    )
    TIME1 = datetime.now()
    duration = TIME1 - TIME0
    print("Calculation took", duration.total_seconds(), "seconds.")
