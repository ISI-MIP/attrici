import os
import numpy as np
import iris
import iris.coord_categorisation as icc
import joblib
from datetime import datetime
import settings as s
import regression

gmt_file = os.path.join(s.data_dir, s.gmt_file)
to_detrend_file = os.path.join(s.data_dir, s.to_detrend_file)

gmt = iris.load_cube(gmt_file)
data_to_detrend = iris.load_cube(to_detrend_file)
icc.add_day_of_year(data_to_detrend, 'time')
doys_cube = data_to_detrend.coord('day_of_year').points

# remove 366th day for now.
data_to_detrend = data_to_detrend[doys_cube != 366]

days_of_year = 365
# interpolate monthly gmt values to daily.
# do this more exact later.
gmt_on_each_day = np.interp(np.arange(110*days_of_year),
                            gmt.coord("time").points,gmt.data)

# lonis = np.arange(data_to_detrend.shape[2])
doys = np.arange(days_of_year)


def run_lat_slice_parallel(lat_slice_data, gmt_on_each_day, days_of_year):

    """ calculate linear regression stats for all days of years and all latitudes.
    joblib implementation. Return a list of all stats """

    lonis = np.arange(lat_slice_data.shape[1])
    doys = np.arange(days_of_year)

    results = joblib.Parallel(n_jobs=3)(
                joblib.delayed(regression.linear_regr_per_gridcell)(
                    lat_slice_data, gmt_on_each_day, doy, loni)
                        for doy in doys for loni in lonis)
    return results


def run_linear_regr_on_iris_cube(cube, days_of_year):

    """ use the iris slicing to run linear regression on a whole iris cube.
    for each latitude slice, calculation is parallelized. """

    results = []
    for lat_slice in cube.slices(['time','longitude']):
        r = run_lat_slice_parallel(lat_slice.data, gmt_on_each_day, days_of_year)
        results = results + r
    return results


### the following functions maybe moved to an "io and iris file" later. ###

def create_doy_cube(array,original_cube_coords, **kwargs):

    """ create an iris cube from a plain numpy array.
    First dimension is always days of the year. Second and third are lat and lon
    and are taken from the input data. """

    doy_coord = iris.coords.DimCoord(np.arange(1.,366.), var_name="day_of_year")

    cube = iris.cube.Cube(array,
            dim_coords_and_dims=[(doy_coord, 0),
             (original_cube_coords('latitude'), 1),
             (original_cube_coords('longitude'), 2),
            ], **kwargs)

    return cube


def write_linear_regression_stats(shape_of_input, original_cube_coords,
        results, file_to_write):

    """ write linear regression statistics to a netcdf file. This function is specific
    to the output of the scipy.stats.linregress output.
    TODO: make this more flexible to include more stats. """

    intercepts = np.zeros([days_of_year, shape_of_input[1],
                       shape_of_input[2]])
    slopes = np.zeros_like(intercepts)

    latis=np.arange(shape_of_input[1])
    lonis=np.arange(shape_of_input[2])

    i = 0
    for lati in latis:
        for doy in np.arange(days_of_year):
            for loni in lonis:
                intercepts[doy,lati,loni] = results[i].intercept
                slopes[doy,lati,loni] = results[i].slope
                i = i + 1

    scube = create_doy_cube(slopes, original_cube_coords, var_name="slope_linear_regr")
    icube = create_doy_cube(intercepts, original_cube_coords, var_name="intercept_linear_regr")

    iris.fileformats.netcdf.save([scube,icube], file_to_write)


if __name__ == "__main__":

    TIME0 = datetime.now()

    results = run_linear_regr_on_iris_cube(data_to_detrend, days_of_year)

    # results = run_parallel_linear_regr(n_jobs=3)
    TIME1 = datetime.now()
    duration = TIME1 - TIME0
    print('Calculation took', duration.total_seconds(), 'seconds.')

    file_to_write = os.path.join(s.data_dir, "testfile.nc4")
    # due to a bug in iris I guess, I cannot overwrite existing files. Remove before.
    if os.path.exists(file_to_write): os.remove(file_to_write)
    write_linear_regression_stats(data_to_detrend.shape, data_to_detrend.coord,
        results, file_to_write)
    TIME2 = datetime.now()
    duration = TIME2 - TIME1
    print('Saving took', duration.total_seconds(), 'seconds.')
