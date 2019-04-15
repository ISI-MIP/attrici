import os
import sys
import datetime
import numpy as np
import re
import warnings
import netCDF4 as nc
from scipy import special
import joblib
import settings as s


def run_lat_slice_parallel(
    lat_slice_data, days_of_year, function_to_run, n_jobs
):

    """ run any function on latitude slices.
        Joblib implementation. Return a list of all stats """

    sys.stdout.flush()
    lonis = np.arange(lat_slice_data.shape[1])
    doys = np.arange(days_of_year)
    TIME0 = datetime.datetime.now()
    results = joblib.Parallel(n_jobs=n_jobs, backend="threading")(
        joblib.delayed(function_to_run.run)(lat_slice_data, doy, loni)
        for doy in doys
        for loni in lonis
    )

    print("Done with slice", flush=True)
    TIME1 = datetime.datetime.now()
    duration = TIME1 - TIME0
    print("Working on slice took", duration.total_seconds(), "seconds.\n", flush=True)
    return results


def run_lat_slice_serial(
    lat_slice_data, gmt_on_each_day, days_of_year, function_to_run
):

    """ run a function on latitude slices.
     Classic serial looping, else it does the same as run_lat_slice_parallel
     Return a list of all stats """

    results = []
    for doy in np.arange(days_of_year):
        for loni in np.arange(lat_slice_data.shape[1]):
            result = function_to_run(lat_slice_data, gmt_on_each_day, doy, loni)
            results.append(result)

    return results


def run_function_on_ncdf(
    data_to_detrend, days_of_year, function_to_run, n_jobs
):

    """ use the numpy slicing to run a function on a full dataset.
    example function: regression.linear_regr_per_gridcell
    for each latitude slice, calculation is parallelized.
    TODO: describe what the function needs to fullfil to be used here.
    """

    i = 0
    results = []
    for lat_slice in np.arange(data_to_detrend.shape[1]):
        data = data_to_detrend[:, i, :]
        print("Working on slice " + str(i), flush=True)
        TIME0 = datetime.datetime.now()
        r = run_lat_slice_parallel(
            data, days_of_year, function_to_run, n_jobs
        )
        results = results + r
        TIME1 = datetime.datetime.now()
        duration = TIME1 - TIME0
        i += 1

    return results


def get_gmt_on_each_day(gmt_file, days_of_year):

    # FIXME: make flexible later
    length_of_record = 110

    ncgmt = nc.Dataset(gmt_file, "r")

    # interpolate from yearly to daily values
    gmt_on_each_day = np.interp(
        np.arange(length_of_record * days_of_year),
        ncgmt.variables["time"][:],
        ncgmt.variables["tas"][:],
    )
    ncgmt.close()

    return gmt_on_each_day


def check_data(data, source_path_data):

    """ This function transforms slices as a
    precprocessing step of data for regression analysis.
    NOTE: add more check routines"""

        # get data folder
    data_folder = os.path.join("/", *source_path_data.split("/")[:-1])

    # get filename
    data_file = source_path_data.split("/")[-1]
    
    var = s.variable

    if isinstance(data, nc.Dataset):
        data = data.variables[var][:]

        # run different transforms for variables
        if var in ["rhs", "hurs"]:
            # print('Data variable is relative humidity!')
            # raise warnings if data contains values that need replacing
            if np.logical_or(data < const.minval[var], 
                             data > const.maxval[var]).any():
                #  warnings.warn('Data contains values far out of range. ' +
                #                'Replacing values by numbers just in range!')
                data[data > const.maxval[var]] = .999999 * const.maxval[var]
                data[data < const.minval[var]] = .000001 * const.maxval[var]
                warnings.warn(
                    "Had to replace out of range values!"
                    + "\nData seems to be erroneous!"
                )
            if np.logical_or(data > 1, data < const.maxval[var]).any():
                data[data > const.maxval[var]] = const.maxval[var]
                warnings.warn("Replaced values of oversaturation!")
            if np.logical_or(data == const.minval[var], data == const.maxval[var]).any():
                #  warnings.warn('Data contains values 0 and/or 100. ' +
                #                '\nReplacing values by numbers just in range!')
                data[data == const.maxval[var]] = 99.9999
                data[data == const.minval[var]] = 0.0001
                warnings.warn("Replaced values bordering open interval (0, 100)!")

        elif variable == "tas":
            print("Data variable is daily mean temperature!", flush=True)

        elif variable == "tasmax":
            print("Data variable is daily maximum temperature!", flush=True)
            tas_file = os.path.join(
                data_folder, re.sub(variable, "tas", source_path_data.split("/")[-1])
            )
            for lat_slice in range(data.shape[1]):
                tas = nc.Dataset(tas_file, "r").variables["tas"][:, lat_slice, :]

                # check values of data

                if np.min(data - tas) <= 0:
                    raise Exception(
                        "\nAt least one value of tasmax appears to be "
                        + "smaller than corresponding tas value!"
                    )

        elif variable == "tasmin":
            print("Data variable is daily minimum temperature!")
            tas_file = os.path.join(
                data_folder, re.sub(variable, "tas", source_path_data.split("/")[-1])
            )
            for lat_slice in range(data.shape[1]):
                tas = nc.Dataset(tas_file, "r").variables["tas"][:, lat_slice, :]

                # check values of data
                if np.min(tas - data) <= 0:
                    raise Exception(
                        "\nAt least one value of tasmin appears to be "
                        + "larger than corresponding tas value!"
                    )

        elif var == "snowfall":
            pr_file = os.path.join(
                data_folder, re.sub(var, "pr", source_path_data.split("/")[-1])
            )
            pr = nc.Dataset(pr_file, "r").variables["pr"][:, lat_slice, :]
            if np.sum(var >= pr) > 0:
                raise Exception(
                    "\nAt least one value of snowfall appears to be "
                    + "larger than corresponding precipitation value!"
                )
        elif var == "pr":
            print("Data variable is daily precipitation!", flush=True)
            if np.min(data) < const.minval[var]:
                data[data < const.minval[var]] = const.minval[var]
                warnings.warn('Set small values to ' + str(const.minval[var]))
        elif var == "ps":
            print("Data variable is near surface pressure!", flush=True)
        elif var == "huss":
            print("Data variable is specific humidity!", flush=True)
            if np.max(data) > const.maxval[var]:
                data[data > const.maxval[var]] = const.maxval[var]
                warnings.warn('Set large values to ' + str(const.maxval[var]))
            if np.min(data) < const.minval[var]:
                data[data < const.minval[var]] = const.minval[var]
                warnings.warn('Set small values to ' + str(const.minval[var]))
        elif var == "rsds":
            print("Data variable is shortwave incoming radiation!", flush=True)
            if np.max(data) > const.maxval[var]:
                data[data > const.maxval[var]] = const.maxval[var]
                warnings.warn('Set large values to ' + str(const.maxval[var]))
        elif var == "rlds":
            print("Data variable is longwave incoming radiation!", flush=True)
            if np.max(data) > const.maxval[var]:
                data[data > const.maxval[var]] = const.maxval[var]
                warnings.warn('Set large values to ' + str(const.maxval[var]))
        elif var == "wind":
            print("Data variable is near surface wind speed!")
            if np.min(data) < const.minval[var]:
                data[data < const.minval[var]] = const.minval[var]
                warnings.warn('Set negative wind speed to ' + str(const.minval[var]))
        else:
            raise Exception("Detected variable name not correct!")
    print("Data checked for forbidden values!", flush=True)
    return data


# FIXME: delete code below?


def create_doy_cube(array, original_cube_coords, **kwargs):

    """ create an iris cube from a plain numpy array.
    First dimension is always days of the year. Second and third are lat and lon
    and are taken from the input data. """

    sys.stdout.flush()
    doy_coord = iris.coords.DimCoord(np.arange(1.0, 366.0), var_name="day_of_year")

    cube = iris.cube.Cube(
        array,
        dim_coords_and_dims=[
            (doy_coord, 0),
            (original_cube_coords("latitude"), 1),
            (original_cube_coords("longitude"), 2),
        ],
        **kwargs
    )

    return cube


def remove_leap_days(data, time):

    """
    removes 366th dayofyear from numpy array that starts on Jan 1st, 1901 with daily timestep
    """

    dates = [datetime(1901, 1, 1) + n * timedelta(days=1) for n in range(time.shape[0])]
    dates = nc.num2date(time[:], units=time.units)
    leap_mask = []
    for date in dates:
        leap_mask.append(date.timetuple().tm_yday != 366)
    return data[leap_mask]

def logit(data):
    if any(data) not in range(const.minval[var], const.maxval[var]+1):
        warnings.warn("Some values seem to be out of range. NaNs are going to be produced!")
    return 2. * np.arctanh(2. * (data - const.minval[var]) / (const.maxval[var] - const.minval[var]) - 1.)

def expit(data):
    return (minval[var] + (maxval[var] - minval[var]) * .5 * (1. + np.tanh(.5 * data)))

def log(data):
    data[data == 0] = np.nan
    return np.log(data)

def exp(data):
    trans = np.exp(data)
    trans[trans == np.nan] = 0
    return trans
