import os
import numpy as np
import netCDF4 as nc
import time
import idetrend.const as c


def get_coefficient_fields(data_path):

    """ get the fields of coefficients of the linear regression
    from netcdf file. """

    ncf = nc.Dataset(data_path, "r")
    slope = ncf.variables["slope"][:]
    intercept = ncf.variables["intercept"][:]
    lat = ncf.variables["lat"][:]
    lon = ncf.variables["lon"][:]
    ncf.close()

    return lat, lon, slope, intercept


def fit_minimal(gmt_on_doy, intercept, slope, transform):

    """ A minimal fit function for 1-dimensional data with trend from regression coefficients"""
    if transform is None:
        fit = intercept + (slope * gmt_on_doy)
    else:
        fit = transform[1](intercept + (slope * gmt_on_doy))

    return fit


class detrending(object):
    def __init__(
        self,
        lons,
        lats,
        slope,
        intercept,
        to_detrend_file,
        variable,
        gmt_on_each_day,
        days_of_year,
    ):

        self.lons = lons
        self.lats = lats
        self.slope = slope
        self.intercept = intercept
        self.to_detrend_file = to_detrend_file
        self.variable = variable
        self.gmt_on_each_day = gmt_on_each_day
        self.days_of_year = days_of_year
        self.transform = c.transform[variable]

    def fit_ts(self, doy, data_to_detrend):

        """ A function to fit 2-dimensional data with trend from regression coefficients.
        Employs fit_minimal()."""

        gmt_on_doy = np.tile(
            self.gmt_on_each_day[doy :: self.days_of_year],
            [data_to_detrend.shape[1], data_to_detrend.shape[2], 1],
        )
        # move time from last to first dimension
        gmt_on_doy = np.moveaxis(gmt_on_doy, -1, 0)

        intercept = self.intercept[doy, :, :]
        slope = self.slope[doy, :, :]

        fit = fit_minimal(gmt_on_doy, intercept, slope, self.transform)
        data_detrended = data_to_detrend - fit + fit[0, :, :]

        return data_detrended

    def get_data_to_detrend(self, doy):

        """ get the timeseries of variable for a specific day of year (doy)
        """

        ncf = nc.Dataset(self.to_detrend_file, "r")
        data_to_detrend = ncf.variables[self.variable][doy :: self.days_of_year, :, :]
        ncf.close()
        return data_to_detrend

    def write_detrended(self, file_to_write):

        """ detrend data and write it to netCDF file. """

        if os.path.exists(file_to_write):
            os.remove(file_to_write)

        # create data set and dimensions
        output_ds = nc.Dataset(file_to_write, "w", format="NETCDF4")

        tm = output_ds.createDimension("time", None)
        lat = output_ds.createDimension("lat", len(self.lats))
        lon = output_ds.createDimension("lon", len(self.lons))
        # print(output_ds.dimensions)

        # create variables
        times = output_ds.createVariable("time", "f8", ("time",))
        longitudes = output_ds.createVariable("lon", "f8", ("lon",))
        latitudes = output_ds.createVariable("lat", "f8", ("lat",))
        data = output_ds.createVariable(self.variable, "f4", ("time", "lat", "lon"))

        # Set attributes
        output_ds.description = "Detrended data of variable " + self.variable
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

        latitudes[:] = self.lats
        longitudes[:] = self.lons
        times[:] = range(len(self.gmt_on_each_day))

        for doy in range(self.days_of_year):
            print("Working on doy: " + str(doy))

            data_to_detrend = self.get_data_to_detrend(doy)
            data[doy :: self.days_of_year, :, :] = self.fit_ts(doy, data_to_detrend)

        output_ds.close()
