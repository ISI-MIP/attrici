#!/usr/bin/env python
# coding: utf-8

from datetime import datetime, timedelta
import joblib
import netCDF4 as nc
import numpy as np
import os
import regression
import settings as s
import sys
import time as t

#  specify paths
#  out_script = '/home/bschmidt/scripts/detrending/output/regr.out'
#  source_path_data = '/home/bschmidt/temp/gswp3/test_data_tas_no_leap.nc4'
#  source_path_gmt = '/home/bschmidt/temp/gswp3/test_ssa_gmt.nc4'
#  dest_path = '/home/bschmidt/temp/gswp3/'

gmt_file = os.path.join(s.data_dir, s.gmt_file)
to_detrend_file = os.path.join(s.data_dir, s.to_detrend_file)

gmt = nc.Dataset(gmt_file, "r")
data = nc.Dataset(to_detrend_file, "r")
var = list(data.variables.keys())[-1]

days_of_year = 365
# interpolate monthly gmt values to daily.
# do this more exact later.
gmt_on_each_day = np.interp(np.arange(110*days_of_year),
                            gmt.variables["time"][:], gmt.variables['gmt'][:])
gmt_on_each_day = gmt.variables['gmt'][:]

def remove_leap_days(data, time):

    '''
    removes 366th dayofyear from numpy array that starts on Jan 1st, 1901 with daily timestep
    '''

    dates = [datetime(1901,1,1)+n*timedelta(days=1) for n in range(time.shape[0])]
    dates = nc.num2date(time[:], units=time.units)
    leap_mask = []
    for date in dates:
        leap_mask.append(date.timetuple().tm_yday != 366)
    return data[leap_mask]


gmt_on_each_day = remove_leap_days(gmt_on_each_day, gmt.variables['time'])
# data_to_detrend = remove_leap_days(data.variables[var], data.variables['time'])
data_to_detrend = data.variables[var]


def run_lat_slice_parallel(lat_slice_data, gmt_on_each_day, days_of_year):

    """ calculate linear regression stats for all days of years and
    all latitudes. Joblib implementation. Return a list of all stats """

    sys.stdout.flush()
    lonis = np.arange(lat_slice_data.shape[1])
    doys = np.arange(days_of_year)
    TIME0 = datetime.now()
    results = joblib.Parallel(n_jobs = s.n_jobs)(
                joblib.delayed(regression.linear_regr_per_gridcell)(
                    lat_slice_data, gmt_on_each_day, doy, loni)
                        for doy in doys for loni in lonis)
    print('Done with slice', flush=True)
    TIME1 = datetime.now()
    duration = TIME1 - TIME0
    print('Working on slice took', duration.total_seconds(), 'seconds.\n', flush=True)
    return results


def run_linear_regr_on_ncdf(data_to_detrend, days_of_year):

    """ use the iris slicing to run linear regression on a whole iris cube.
    for each latitude slice, calculation is parallelized. """
    i = 0
    results = []
    for lat_slice in np.arange(data_to_detrend.shape[1]):  #  variables[var].
        data = data_to_detrend[:, i, :]
        print('Working on slice ' + str(i), flush=True)
        TIME0 = datetime.now()
        r = run_lat_slice_parallel(data, gmt_on_each_day, days_of_year)
        results = results + r
        TIME1 = datetime.now()
        duration = TIME1 - TIME0
        i += 1
    return results


# the following functions maybe moved to an "io file" later. ###

def create_doy_cube(array, original_cube_coords, **kwargs):

    """ create an iris cube from a plain numpy array.
    First dimension is always days of the year. Second and third are lat and lon
    and are taken from the input data. """

    sys.stdout.flush()
    doy_coord = iris.coords.DimCoord(np.arange(1., 366.), var_name="day_of_year")

    cube = iris.cube.Cube(array,
            dim_coords_and_dims=[(doy_coord, 0),
             (original_cube_coords('latitude'), 1),
             (original_cube_coords('longitude'), 2),
            ], **kwargs)

    return cube


def write_linear_regression_stats(shape_of_input, original_data_coords,
        results, file_to_write):

    """ write linear regression statistics to a netcdf file. This function is specific
    to the output of the scipy.stats.linregress output.
    TODO: make this more flexible to include more stats. """

    sys.stdout.flush()
    output_ds = nc.Dataset(file_to_write, "w", format="NETCDF4")
    time = output_ds.createDimension("time", None)
    lat = output_ds.createDimension("lat", original_data_coords[0].shape[0])
    lon = output_ds.createDimension("lon", original_data_coords[1].shape[0])
    print(output_ds.dimensions)
    times = output_ds.createVariable("time", "f8", ("time",))
    longitudes = output_ds.createVariable("lon", "f4", ("lon",))
    latitudes = output_ds.createVariable("lat", "f4", ("lat",))
    intercepts = output_ds.createVariable('intercept', "f8", ("time", "lat", "lon",))
    slopes = output_ds.createVariable('slope', "f8", ("time", "lat", "lon",))
    print(intercepts)

    output_ds.description = "Regression test script"
    output_ds.history = "Created " + t.ctime(t.time())
    latitudes.units = "degrees north"
    longitudes.units = "degrees east"
    slopes.units = "K"
    intercepts.units = "K"
    times.units = "days since 01-01 00:00:00.0"
    times.calendar = "365 days"

    lats = original_data_coords[0][:]
    lons = original_data_coords[1][:]

    latitudes[:] = lats
    longitudes[:] = lons

    print("latitudes: \n", latitudes[:])
    print("longitudes: \n", longitudes[:])

    ic = np.zeros([days_of_year, shape_of_input[1],
                   shape_of_input[2]])
    s = np.zeros_like(ic)

    latis = np.arange(shape_of_input[1])
    lonis = np.arange(shape_of_input[2])

    i = 0
    for lati in latis:
        for doy in np.arange(days_of_year):
            for loni in lonis:
                ic[doy, lati, loni] = results[i].intercept
                s[doy, lati, loni] = results[i].slope
                i = i + 1
    intercepts[:] = ic
    slopes[:] = s
    output_ds.close()


if __name__ == "__main__":

    TIME0 = datetime.now()

    results = run_linear_regr_on_ncdf(data_to_detrend, days_of_year)

    # results = run_parallel_linear_regr(n_jobs=3)
    TIME1 = datetime.now()
    duration = TIME1 - TIME0
    print('Calculation took', duration.total_seconds(), 'seconds.')

    file_to_write = os.path.join(s.data_dir, "first_real_test.nc4")
    # due to a bug in iris I guess, I cannot overwrite existing files. Remove before.
    if os.path.exists(file_to_write): os.remove(file_to_write)
    write_linear_regression_stats(data_to_detrend.shape,
                                  (data.variables['lat'],
                                   data.variables['lon']),
                                  results, file_to_write)
    TIME2 = datetime.now()
    duration = TIME2 - TIME1
    print('Saving took', duration.total_seconds(), 'seconds.')
