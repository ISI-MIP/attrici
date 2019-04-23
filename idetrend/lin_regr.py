import os
import sys
import time
from datetime import datetime

# import iris
# import iris.coord_categorisation as icc
import numpy as np
import netCDF4 as nc
import settings as s
from scipy.stats import mstats
from collections import namedtuple
import idetrend.const as c


class regression(object):
    def __init__(self, gmt_on_each_day, min_ts_len, transform=None):

        self.gmt_on_each_day = gmt_on_each_day
        self.transform = transform
        self.min_ts_len = min_ts_len

        # FIXME: use this once custom transforms are not needed anymore
        # currently only works on numpy arrays.
        # assert 0.3 == transform[1](transform[0](.3)), "Inverse transform does not match with transform."

    def run(self, np_data_to_detrend, doy, loni=0):

        """ minimal version of a linear regression per grid cell """

        if np_data_to_detrend.ndim >= 2:
            data_of_doy = np_data_to_detrend[doy::365, loni]
        else:
            data_of_doy = np_data_to_detrend[doy::365]

        # FIXME: we here select gmt that is interpolated from yearly values,
        # so gmt_of_doy depends on gmt[i] and gmt[i+1] or so. do we want that?
        # would it not be better to pass yearly GMT here?
        gmt_of_doy = self.gmt_on_each_day[doy::365]

        # special case if too few valid datapoints left
        if data_of_doy.count() <= self.min_ts_len:
            res = namedtuple(
                "LinregressResult", ("slope", "intercept", "rvalue", "pvalue", "stderr")
            )
            return res(slope=0.0, intercept=0.0, rvalue=0.0, pvalue=0.0, stderr=0.0)
            # data_of_doy.mask = True

        if self.transform is not None:
            data_of_doy = self.transform[0](data_of_doy)

        # print(data_of_doy)
        return mstats.linregress(gmt_of_doy, data_of_doy)


def write_regression_stats(
    shape_of_input, original_data_coords, results, file_to_write, days_of_year
):

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
    intercepts = output_ds.createVariable("intercept", "f8", ("time", "lat", "lon"))
    slopes = output_ds.createVariable("slope", "f8", ("time", "lat", "lon"))
    r_values = output_ds.createVariable("r_values", "f8", ("time", "lat", "lon"))
    p_values = output_ds.createVariable("p_values", "f8", ("time", "lat", "lon"))
    std_errors = output_ds.createVariable("std_errors", "f8", ("time", "lat", "lon"))
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

    ic = np.ma.masked_all([days_of_year, shape_of_input[1], shape_of_input[2]])
    s = np.ma.copy(ic)
    r = np.ma.copy(ic)
    p = np.ma.copy(ic)
    sd = np.ma.copy(ic)

    latis = np.arange(shape_of_input[1])
    lonis = np.arange(shape_of_input[2])

    i = 0
    for lati in latis:
        for doy in np.arange(days_of_year):
            for loni in lonis:
                # try:
                ic[doy, lati, loni] = results[i].intercept
                s[doy, lati, loni] = results[i].slope
                r[doy, lati, loni] = results[i].rvalue
                p[doy, lati, loni] = results[i].pvalue
                sd[doy, lati, loni] = results[i].stderr
                # if not enough valid values appeared in timeseries,
                # regression gives back Nones, catch this here.
                # except AttributeError:
                #     print(lati,doy,loni)
                #     print(results[i])
                #     pass
                i = i + 1

    intercepts[:] = ic
    slopes[:] = s
    r_values[:] = r
    p_values[:] = p
    std_errors[:] = sd
    output_ds.close()
