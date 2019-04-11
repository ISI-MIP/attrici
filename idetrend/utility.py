import os
import sys
import datetime
import numpy as np
import re
import warnings
import netCDF4 as nc
from scipy import special
import joblib


def run_lat_slice_parallel(lat_slice_data, gmt_on_each_day, days_of_year,
                           function_to_run, n_jobs):

    """ run any function on latitude slices.
        Joblib implementation. Return a list of all stats """

    sys.stdout.flush()
    lonis = np.arange(lat_slice_data.shape[1])
    doys = np.arange(days_of_year)
    TIME0 = datetime.datetime.now()
    results = joblib.Parallel(n_jobs=n_jobs, backend='threading')(
                joblib.delayed(function_to_run)(
                    lat_slice_data, gmt_on_each_day, doy, loni)
                        for doy in doys for loni in lonis)

    print('Done with slice', flush=True)
    TIME1 = datetime.datetime.now()
    duration = TIME1 - TIME0
    print('Working on slice took', duration.total_seconds(), 'seconds.\n', flush=True)
    return results


def run_function_on_ncdf(data_to_detrend, gmt_on_each_day,
        days_of_year, function_to_run, n_jobs):

    """ use the numpy slicing to run a function on a full dataset.
    example function: regression.linear_regr_per_gridcell
    for each latitude slice, calculation is parallelized.
    TODO: describe what the function needs to fullfil to be used here.
    """

    i = 0
    results = []
    for lat_slice in np.arange(data_to_detrend.shape[1]):
        data = data_to_detrend[:, i, :]
        print('Working on slice ' + str(i), flush=True)
        TIME0 = datetime.datetime.now()
        r = run_lat_slice_parallel(data, gmt_on_each_day, days_of_year,
            function_to_run, n_jobs)
        results = results + r
        TIME1 = datetime.datetime.now()
        duration = TIME1 - TIME0
        i += 1

    return results


def get_gmt_on_each_day(gmt_file, days_of_year):

    # FIXME: make flexible later
    length_of_record = 110

    gmt = nc.Dataset(gmt_file, "r")
    # gmt_var = list(gmt.variables.keys())[-1]
    # print(gmt_var, flush=True)
    gmt_on_each_day = np.interp(
        np.arange(length_of_record*days_of_year),
        gmt.variables["time"][:], gmt.variables["tas"][:])
    return gmt_on_each_day
    # print(var, flush=True)


def check_data(data, source_path_data):

    """ This function transforms slices as a
    precprocessing step of data for regression analysis.
    NOTE: add more check routines"""

    # get variable name from filename
    var_list = ['rhs', 'hurs', 'huss', 'snowfall', 'ps', 'rlds',
                'rsds', 'wind', 'tasmin', 'tasmax', 'tas']
    # get data folder
    data_folder = os.path.join('/', *source_path_data.split('/')[:-1])

    # fet filename
    data_file = source_path_data.split('/')[-1]

    # get variable (tas needs to be last in var_list to avoid wrong detection
    # when tasmax or tasmin are searched
    variable = [variable for variable in var_list if variable in data_file][0]

    if isinstance(data, nc.Dataset):
        data = data.variables['rhs'][:]

    for lat_slice in range(data.shape[1]):
        # run different transforms for variables
        if variable in ['rhs', 'hurs']:
            # print('Data variable is relative humidity!')
            # raise warnings if data contains values that need replacing
            if np.sum(np.logical_or(data < 0, data > 110)) > 0:
                #  warnings.warn('Data contains values far out of range. ' +
                #                'Replacing values by numbers just in range!')
                data[data > 110] = 99.9999
                data[data < 0] = .0001
                warnings.warn('Had to replace values that are far out of range!' +
                              '\nData seems to be erroneous!')
            if np.sum(np.logical_or(data > 100, data < 110)) > 0:
                #  warnings.warn('Some values suggest oversaturation. ' +
                              #  'Replacing values by numbers just in range!')
                data[data > 100] = 99.9999
                warnings.warn('Replaced values of oversaturation!')
            if np.sum(np.logical_or(data == 0, data == 100)) > 0:
                #  warnings.warn('Data contains values 0 and/or 100. ' +
                #                '\nReplacing values by numbers just in range!')
                data[data == 100] = 99.9999
                data[data == 0] = .0001
                warnings.warn('Replaced values bordering open interval (0, 100)!')

        elif variable == 'tas':
            print('Data variable is daily mean temperature!')

        elif variable == 'tasmax':
            print('Data variable is daily maximum temperature!')
            tas_file = os.path.join(data_folder,
                                    re.sub(variable,
                                           'tas',
                                           source_path_data.split('/')[-1]))
            tas = nc.Dataset(tas_file, "r").variables['tas'][:, lat_slice, :]

            # check values of data

            if np.min(data - tas) <= 0:
                raise Exception('\nAt least one value of tasmax appears to be ' +
                                'smaller than corresponding tas value!')

        elif variable == 'tasmin':
            print('Data variable is daily minimum temperature!')
            tas_file = os.path.join(data_folder,
                                    re.sub(variable,
                                           'tas',
                                           source_path_data.split('/')[-1]))
            tas = nc.Dataset(tas_file, "r").variables['tas'][:, lat_slice, :]

            # check values of data
            if np.min(tas - data) <= 0:
                raise Exception('\nAt least one value of tasmin appears to be ' +
                                'larger than corresponding tas value!')

        elif variable == 'snowfall':
            pr_file = os.path.join(data_folder,
                                   re.sub(variable,
                                          'pr',
                                          source_path_data.split('/')[-1]))
            pr = nc.Dataset(pr_file, "r").variables['pr'][:, lat_slice, :]
            if np.sum(variable >= pr) > 0:
                raise Exception('\nAt least one value of snowfall appears to be ' +
                                'larger than corresponding precipitation value!')
        elif variable == 'pr':
            print('Data variable is daily precipitation!')
        elif variable == 'ps':
            print('Data variable is near surface pressure!')
        elif variable == 'huss':
            print('Data variable is specific humidity!')
        elif variable == 'rsds':
            print('Data variable is shortwave incoming radiation!')
        elif variable == 'rlds':
            print('Data variable is longwave incoming radiation!')
        elif variable == 'wind':
            print('Data variable is near surface wind speed!')
        else:
            raise Exception('Detected variable name not correct!')
    print('Data checked for forbidden values!', flush=True)
    return data


# FIXME: delete code below?
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

