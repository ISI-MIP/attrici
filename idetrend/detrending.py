import os
import numpy as np
import netCDF4 as nc
import time
import idetrend.const as c
import idetrend.visualization as vis

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
    doy,
    transform=None,
    calendar="noleap",
):

    """ A function to fit 2-dimensional data with trend from regression coefficients.
    Employs fit_minimal()."""

    if calendar == "noleap":
        days_of_year = 365

    lat, lon, slope, intercept = vis.get_regression_coefficients(regr_path, [doy])
    data_to_detrend = vis.get_data_to_detrend(data_path, variable, [doy])
    gmt_on_doy = np.tile(
        gmt_on_each_day[doy::days_of_year],
        [data_to_detrend.shape[1], data_to_detrend.shape[2], 1],
    )
    # move time from last to first dimension
    gmt_on_doy = np.moveaxis(gmt_on_doy, -1, 0)

    fit = fit_minimal(gmt_on_doy, intercept, slope, transform)
    data_detrended = data_to_detrend - fit + fit[0, :, :]

    return data_detrended


def write_detrended(
    regr_path, data_path, lats, lons, file_to_write, variable, gmt_on_each_day
):

    """ datrend data and write it to netCDF file. """

    if os.path.exists(file_to_write):
        os.remove(file_to_write)

    # create data set and dimensions
    output_ds = nc.Dataset(file_to_write, "w", format="NETCDF4")

    tm = output_ds.createDimension("time", None)
    lat = output_ds.createDimension("lat", len(lats))
    lon = output_ds.createDimension("lon", len(lons))
    # print(output_ds.dimensions)

    # create variables
    times = output_ds.createVariable("time", "f8", ("time",))
    longitudes = output_ds.createVariable("lon", "f8", ("lon",))
    latitudes = output_ds.createVariable("lat", "f8", ("lat",))
    data = output_ds.createVariable(variable, "f4", ("time", "lat", "lon"))
    # print(intercepts)

    # Set attributes
    output_ds.description = "Detrended data of variable " + variable
    output_ds.history = "Created " + time.ctime(time.time())
    latitudes.units = "degree_north"
    latitudes.long_name = "latitude"
    longitudes.standard_name = "latitude"
    longitudes.units = "degree_east"
    longitudes.long_name = "longitude"
    longitudes.standard_name = "longitude"
    data.units = "K"
    times.units = "days since 1901-01-01"
    times.calendar = "noleap"

    if times.calendar == "noleap":
        days_of_year = 365
        # doys = range(1, 366)

    latitudes[:] = lats
    longitudes[:] = lons
    times[:] = range(len(gmt_on_each_day))

    #  latis = np.arange(shape_of_input[1])
    #  lonis = np.arange(shape_of_input[2])

    for doy in range(days_of_year):
        # print("Working on doy: " + str(doy))
        data_detrended = fit_ts(
            regr_path,
            data_path,
            variable,
            gmt_on_each_day,
            doy,
            transform=c.transform[variable],
        )
        data[doy::days_of_year, :, :] = data_detrended
    output_ds.close()
