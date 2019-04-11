import os
import sys
import time
from datetime import datetime
import iris
import iris.coord_categorisation as icc
import numpy as np
import netCDF4 as nc
import settings as s
from scipy import stats


def regr_per_gridcell(np_data_to_detrend, gmt_on_each_day, doy, loni=0):

    """ minimal version of a linear regression per grid cell """
    #TIME1 = datetime.now()
    # print(np_data_to_detrend.shape, flush=True)
    # print(gmt_on_each_day.shape, flush=True)
    #print('doy is: ' + str(doy), flush=True)
    #print('longitude index is: ' + str(loni), flush=True)

    if np_data_to_detrend.ndim >= 2:
        data_of_doy = np_data_to_detrend[doy::365, loni]
    else:
        data_of_doy = np_data_to_detrend[doy::365]

    gmt_of_doy = gmt_on_each_day[doy::365]
    #TIME2 = datetime.now()
    #duration = TIME2 - TIME1
    #print('One grid cell regression input took', duration.total_seconds(), 'seconds.', flush=True)
    return stats.linregress(gmt_of_doy, data_of_doy)


def write_regression_stats(shape_of_input, original_data_coords,
        results, file_to_write, days_of_year):

    """ write linear regression statistics to a netcdf file. This function is specific
    to the output of the scipy.stats.linregress output.
    TODO: make this more flexible to include more stats. """

    sys.stdout.flush()
    output_ds = nc.Dataset(file_to_write, "w", format="NETCDF4")
    tm = output_ds.createDimension("time", None)
    lat = output_ds.createDimension("lat", original_data_coords[0].shape[0])
    lon = output_ds.createDimension("lon", original_data_coords[1].shape[0])
    print(output_ds.dimensions)
    times = output_ds.createVariable("time", "f8", ("time",))
    longitudes = output_ds.createVariable("lon", "f4", ("lon",))
    latitudes = output_ds.createVariable("lat", "f4", ("lat",))
    intercepts = output_ds.createVariable('intercept', "f8", ("time", "lat", "lon",))
    slopes = output_ds.createVariable('slope', "f8", ("time", "lat", "lon",))
    r_values = output_ds.createVariable('r_values', "f8", ("time", "lat", "lon",))
    p_values = output_ds.createVariable('p_values', "f8", ("time", "lat", "lon",))
    std_errors = output_ds.createVariable('std_errors', "f8", ("time", "lat", "lon",))
    print(intercepts)

    output_ds.description = "Regression test script"
    output_ds.history = "Created " + time.ctime(time.time())
    latitudes.units = "degrees north"
    longitudes.units = "degrees east"
    slopes.units = ""
    intercepts.units = ""
    times.units = "days since 1901-01-01 00:00:00.0"
    times.calendar = "365_day"

    lats = original_data_coords[0][:]
    lons = original_data_coords[1][:]

    latitudes[:] = lats
    longitudes[:] = lons

    print("latitudes: \n", latitudes[:])
    print("longitudes: \n", longitudes[:])

    ic = np.zeros([days_of_year, shape_of_input[1],
                   shape_of_input[2]])
    s = np.zeros_like(ic)
    r = np.zeros_like(ic)
    p = np.zeros_like(ic)
    sd = np.zeros_like(ic)

    latis = np.arange(shape_of_input[1])
    lonis = np.arange(shape_of_input[2])

    i = 0
    for lati in latis:
        for doy in np.arange(days_of_year):
            for loni in lonis:
                ic[doy, lati, loni] = results[i].intercept
                s[doy, lati, loni] = results[i].slope
                r[doy, lati, loni] = results[i].rvalue
                p[doy, lati, loni] = results[i].pvalue
                sd[doy, lati, loni] = results[i].stderr
                i = i + 1
    intercepts[:] = ic
    slopes[:] = s
    r_values[:] = r
    p_values[:] = p
    std_errors[:] = sd
    output_ds.close()


def run_linear_regr_on_iris_cube(cube, days_of_year):

    """ use the iris slicing to run linear regression on a whole iris cube.
    for each latitude slice, calculation is parallelized. """

    results = []
    TIME0 = datetime.now()
    for doy in np.arange(days_of_year):
        print('\nDay of Year is: ' + str(doy), flush=True)
        r = linear_regr_per_gridcell(cube.data, gmt_on_each_day, days_of_year, 0)
        results.append(r)
    TIME1 = datetime.now()
    duration = TIME1 - TIME0
    print('Calculation took', duration.total_seconds(), 'seconds.', flush=True)
    sys.stdout.flush()
    return results


if __name__ == "__main__":

    """ add a quick test for one grid cell of our regression algorithm here. """

    # FIXME: make this compliant with other code. Now this relies on iris,
    # though we have moved on in our other code.
    # probably better done in a tests folder through unit testing.

    gmt_file = os.path.join(s.data_dir, s.gmt_file)
    to_detrend_file = os.path.join(s.data_dir, s.to_detrend_file)

    gmt = iris.load_cube(gmt_file)
    data_to_detrend = iris.load_cube(to_detrend_file)
    icc.add_day_of_year(data_to_detrend, 'time')
    data_to_detrend = data_to_detrend.extract(iris.Constraint(latitude=52.25, longitude=13.25))
    doys_cube = data_to_detrend.coord('day_of_year').points

    # remove 366th day for now.
    data_to_detrend = data_to_detrend[doys_cube != 366]

    days_of_year = 365
    # interpolate monthly gmt values to daily.
    # do this more exact later.
    gmt_on_each_day = np.interp(np.arange(110*days_of_year),
                                gmt.coord("time").points, gmt.data)

    # lonis = np.arange(data_to_detrend.shape[2])
    doys = np.arange(days_of_year)
    print('Starting with regression calculation')

    TIME0 = datetime.now()

    results = run_linear_regr_on_iris_cube(data_to_detrend, days_of_year)

    print('\nThis is the result:\n')
    print(results)
    TIME1 = datetime.now()
    duration = TIME1 - TIME0
    print('Calculation took', duration.total_seconds(), 'seconds.', flush=True)
    # results = run_parallel_linear_regr(n_jobs=3)
