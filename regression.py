from datetime import datetime
import iris
import iris.coord_categorisation as icc
import numpy as np
import os
import run_regression_classic as rr
import settings as s
from scipy import stats
import sys


def linear_regr_per_gridcell(np_data_to_detrend, gmt_on_each_day, doy, loni=0):

    """ minimal version of a linear regression per grid cell """
    #TIME1 = datetime.now()
    #print(np_data_to_detrend.shape, flush=True)
    #print(gmt_on_each_day.shape, flush=True)
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


def run_lat_slice_serial(lat_slice_data, gmt_on_each_day, days_of_year):

    """ calculate linear regression stats for all days of years and
    all grid cells. Classic serial looping. Return a list of all stats """

    results = []
    for doy in np.arange(days_of_year):
        for loni in np.arange(lat_slice_data.shape[1]):
            result = linear_regr_per_gridcell(
                lat_slice_data, gmt_on_each_day, doy, loni)
            results.append(result)

    return results

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
