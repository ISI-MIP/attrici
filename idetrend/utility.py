import os
import sys
import datetime
import numpy as np
import re
import warnings
import netCDF4 as nc

#  from scipy import special
import joblib

#  import idetrend.const
import settings as s


def run_lat_slice_parallel(lat_slice_data, days_of_year, function_to_run, n_jobs):

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
            result = function_to_run.run(lat_slice_data, gmt_on_each_day, doy, loni)
            results.append(result)

    return results


def run_regression_on_dataset(data_to_detrend, days_of_year, function_to_run, n_jobs):

    """ use the numpy slicing to run a function on a full dataset.
    example function: regression.linear_regr_per_gridcell
    for each latitude slice, calculation is parallelized.
    TODO: describe what the function needs to fullfil to be used here.
    """

    i = 0
    results = []
    for lat_slice in np.arange(data_to_detrend.shape[1]):

        data = data_to_detrend[:, i, :]
        # data = mask_invalid(data_to_detrend[:, i, :], minval, maxval)
        print("Working on slice " + str(i), flush=True)
        TIME0 = datetime.datetime.now()
        r = run_lat_slice_parallel(data, days_of_year, function_to_run, n_jobs)
        results = results + r
        TIME1 = datetime.datetime.now()
        duration = TIME1 - TIME0
        i += 1
        # if i >3: break

    return results


def get_gmt_on_each_day(gmt_file, days_of_year):

    length_of_record = s.endyear - s.startyear + 1
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

    from settings import variable

    # get data folder
    data_folder = os.path.join("/", *source_path_data.split("/")[:-1])

    # get filename
    data_file = source_path_data.split("/")[-1]

    if isinstance(data, nc.Dataset):
        data = data.variables[variable][:]

        # run different transforms for variables
        if variable in ["rhs", "hurs"]:
            # print('Data variable is relative humidity!')
            # raise warnings if data contains values that need replacing
            if np.logical_or(
                data < const.minval[variable], data > const.maxval[variable]
            ).any():
                #  warnings.warn('Data contains values far out of range. ' +
                #                'Replacing values by numbers just in range!')
                data[data > const.maxval[variable]] = 0.999999 * const.maxval[variable]
                data[data < const.minval[variable]] = 0.000001 * const.maxval[variable]
                warnings.warn(
                    "Had to replace out of range values!"
                    + "\nData seems to be erroneous!"
                )
            if np.logical_or(data > 1, data < const.maxval[variable]).any():
                data[data > const.maxval[variable]] = const.maxval[variable]
                warnings.warn("Replaced values of oversaturation!")
            if np.logical_or(
                data == const.minval[variable], data == const.maxval[variable]
            ).any():
                #  warnings.warn('Data contains values 0 and/or 100. ' +
                #                '\nReplacing values by numbers just in range!')
                data[data == const.maxval[variable]] = 99.9999
                data[data == const.minval[variable]] = 0.0001
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

        elif variable == "snowfall":
            pr_file = os.path.join(
                data_folder, re.sub(variable, "pr", source_path_data.split("/")[-1])
            )
            pr = nc.Dataset(pr_file, "r").variables["pr"][:, lat_slice, :]
            if np.sum(variable >= pr) > 0:
                raise Exception(
                    "\nAt least one value of snowfall appears to be "
                    + "larger than corresponding precipitation value!"
                )
        elif variable == "pr":
            print("Data variable is daily precipitation!", flush=True)
            if np.min(data) < const.minval[variable]:
                data[data < const.minval[variable]] = const.minval[variable]
                warnings.warn("Set small values to " + str(const.minval[variable]))
        elif variable == "ps":
            print("Data variable is near surface pressure!", flush=True)
        elif variable == "huss":
            print("Data variable is specific humidity!", flush=True)
            if np.max(data) > const.maxval[variable]:
                data[data > const.maxval[variable]] = const.maxval[variable]
                warnings.warn("Set large values to " + str(const.maxval[variable]))
            if np.min(data) < const.minval[variable]:
                data[data < const.minval[variable]] = const.minval[variable]
                warnings.warn("Set small values to " + str(const.minval[variable]))
        elif variable == "rsds":
            print("Data variable is shortwave incoming radiation!", flush=True)
            if np.max(data) > const.maxval[variable]:
                data[data > const.maxval[variable]] = const.maxval[variable]
                warnings.warn("Set large values to " + str(const.maxval[variable]))
        elif variable == "rlds":
            print("Data variable is longwave incoming radiation!", flush=True)
            if np.max(data) > const.maxval[variable]:
                data[data > const.maxval[variable]] = const.maxval[variable]
                warnings.warn("Set large values to " + str(const.maxval[variable]))
        elif variable == "wind":
            print("Data variable is near surface wind speed!")
            if np.min(data) < const.minval[variable]:
                data[data < const.minval[variable]] = const.minval[variable]
                warnings.warn(
                    "Set negative wind speed to " + str(const.minval[variable])
                )
        else:
            raise Exception("Detected variable name not correct!")
    print("Data checked for forbidden values!", flush=True)
    return data

def copy_nc_container(ds, data):
    """ Creates a new netCDF file with dimensions and variables as in data"""

    for name, dimension in data.dimensions.items():
        ds.createDimension(name, len(dimension) if not dimension.isunlimited() else None)

    for name, variable in data.variables.items():
        x = ds.createVariable(name, variable.datatype, variable.dimensions)
    ds.variables["time"][:] = data.variables["time"][:]
    ds.variables["time"].units = data.variables["time"].units
    ds.variables["lat"][:] = data.variables["lat"][:]
    ds.variables["lon"][:] = data.variables["lon"][:]

def y_inv(y, y_orig):
    """rescale data y to y_original"""
    return y * (y_orig.max() - y_orig.min()) + y_orig.min()
# def logit(data):
#     from settings import variable

#     if np.any(data) not in range(
#         int(const.minval[variable]), int(const.maxval[variable] + 1)
#     ):
#         warnings.warn(
#             "Some values seem to be out of range. NaNs are going to be produced!"
#         )
#     return 2.0 * np.arctanh(
#         2.0
#         * (data - const.minval[variable])
#         / (const.maxval[variable] - const.minval[variable])
#         - 1.0
#     )


# def expit(data):
#     from settings import variable

#     if np.any(data) <= 0:
#         warnings.warn("Some values negative or zero. NaNs are going to be produced!")
#     return const.minval[variable] + (
#         const.maxval[variable] - const.minval[variable]
#     ) * 0.5 * (1.0 + np.tanh(0.5 * data))


# def log(data):
#     data[data <= 0] = np.nan
#     return np.log(data)


# def exp(data):
#     trans = np.exp(data)
#     trans[trans == np.nan] = 0
#     return trans
