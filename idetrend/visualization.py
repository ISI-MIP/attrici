#!/usr/bin/env python3
# coding: utf-8

from __future__ import absolute_import, division, print_function
from six.moves import filter, input, map, range, zip  # noqa
from scipy import stats
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import os
import settings as set
import netCDF4 as nc
import const

# fixed numbers for now.
days_of_year = 365
years_of_data = 110  # from 1900 to 2010
doy = np.arange(days_of_year)
time_values = np.arange(years_of_data) + 1901


def set_ylim(ax, data):

    """ manually set ylimit. needed to for plt.scatter, which has
    a bug for setting ylim for small values. """

    dy = (max(data) - min(data)) / 20.0
    ax.set_ylim(min(data) - dy, max(data) + dy)


def get_gmt_on_each_day(gmt_data_path):

    """ interpolate from yearly to daily values
    FIXME: use same function from utility.py and delete here."""
    gmt_data = nc.Dataset(gmt_data_path, "r")
    gmt_on_each_day = np.interp(
        np.arange(years_of_data * days_of_year),
        gmt_data.variables["time"][:],
        gmt_data.variables["tas"][:],
    )
    gmt_data.close()
    return gmt_on_each_day


def get_regression_coefficients(data_path, indices):

    """ get the coefficients of the linear regression
    for a specific day of year, lon and lat from netcdf file. """

    print(indices)
    ncf = nc.Dataset(data_path, "r")
    if len(indices) == 3:
        lat = ncf.variables['lat'][indices[1]]
        lon = ncf.variables['lon'][indices[2]]
    else:
        lat = ncf.variables['lat'][:]
        lon = ncf.variables['lon'][:]

    slope = ncf.variables['slope'][indices]
    intercept = ncf.variables['intercept'][indices]
    se = ncf.variables['std_errors'][indices]
    ncf.close()

    return lat, lon, slope, intercept, se


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


def get_data_to_detrend(data_path, varname, indices):

    """ get the timeseries of variable for a specific day of year (doy),
    lon and lat from netcdf file. doy, lon, lat is from indices. """

    nc_data_to_detrend = nc.Dataset(data_path, "r")
    if len(indices) == 3:
        data_to_detrend = nc_data_to_detrend.variables[varname][
            indices[0]::days_of_year, indices[1], indices[2]]
    else:
        data_to_detrend = nc_data_to_detrend.variables[varname][
            indices[0]::days_of_year, :, :]
    nc_data_to_detrend.close()
    return data_to_detrend

def prepare(regr_path, data_path, variable, gmt_on_each_day, indices, transform=None):

    """ detrend the data with linear trend from regression coefficients.
        also do a regression on detrended data and check if slope is close to zero."""

    from write_detrended_data import fit_minimal
    lat, lon, slope, intercept, se = get_regression_coefficients(regr_path, indices)
    data_to_detrend = get_data_to_detrend(data_path, variable, indices)

    gmt_on_doy = gmt_on_each_day[indices[0] :: days_of_year]
    fit = fit_minimal(gmt_on_doy, intercept, slope, transform)
    data_detrended = data_to_detrend - fit + fit[0]

    if const.minval[variable] is not None:
        data_detrended[data_detrended < const.minval[variable]] = const.minval[variable]
        data_detrended_for_reg = data_detrended[data_detrended > const.minval[variable]]
        gmt_test_for_reg = gmt_on_doy[data_detrended > const.minval[variable]]
    if const.maxval[variable] is not None:
        data_detrended[data_detrended < const.maxval[variable]] = const.maxval[variable]
    if transform is not None:
        # redo a fit on the detrended data. should have a slope of zero
        slope_d, intercept_d, r, p, sd = stats.linregress(gmt_on_doy,
                                                          set.transform[set.variable][0](data_detrended))
    else:
        slope_d, intercept_d, r, p, sd = stats.linregress(gmt_on_doy, data_detrended)
    # assert(abs(slope_d) < 1e-10), ("Slope in detrended data is",abs(slope_d),"and not close to zero.")
    fit_d = fit_minimal(gmt_on_doy, intercept_d, slope_d, transform)

    return data_to_detrend, data_detrended, fit, fit_d, gmt_on_doy


def plot(varname, data_to_detrend, data_detrended, fit, fit_d, gmt_on_doy, lat, lon):

    fig, axs = plt.subplots(2, 2, sharey="row", figsize=(16, 12))
    #     fig.suptitle('var:' + variable + '   doy:' + str(indices[0]) +
    #                  '   lat:' + str(lat) + '   lon:' + str(lon), size=20, weight='bold')

    axs[0, 0].scatter(gmt_on_doy, data_to_detrend, label="data")
    axs[0, 0].plot(gmt_on_doy, fit, "r", label="fit against gmt")
    axs[0, 0].set_xlabel("gmt / K")
    set_ylim(axs[0, 0], data_to_detrend)

    axs[1, 0].scatter(gmt_on_doy, data_detrended, label="detrended data")
    axs[1, 0].plot(gmt_on_doy, fit_d, "r", label="detrended fit")
    axs[1, 0].set_xlabel("gmt / K")
    set_ylim(axs[1, 0], data_detrended)

    axs[0, 1].scatter(time_values, data_to_detrend, label="data")
    axs[0, 1].plot(time_values, fit, "r", label="gmt(t) * fit")
    axs[0, 1].set_xlabel("Years")

    axs[1, 1].scatter(time_values, data_detrended, label="detrended data")
    axs[1, 1].plot(time_values, fit_d, "r", label="gmt(t) * detrended fit")
    axs[1, 1].set_xlabel("Years")

    #  axs[2, 0].plot(slope[lat, lon])
    #  axs[2, 0].set_xlabel('Day of Year')
    #  axs[2, 0].set_ylabel('slope')
    #  axs[2, 0].grid()
    #
    #  axs[2, 1].plot(intercept[lat, lon])
    #  axs[2, 1].set_xlabel('Day of Year')
    #  axs[2, 1].set_ylabel('intercept')
    #  axs[2, 1].grid()

    for ax in axs.ravel():
        ax.grid()
        ax.legend(ncol=1)
        ax.set_ylabel(varname + " in [unit]")


def plot_map(
    variable,
    coeff_name,
    day_of_year,
    varname,
    lat,
    lon,
    cross=None,
    circle=None,
    **kwargs
):

    plt.figure(figsize=(16, 10))
    ax = plt.subplot(111, projection=ccrs.PlateCarree(central_longitude=0))
    variable_at_doy = variable[day_of_year, :, :]
    # ab = np.max(np.abs(variable_at_doy))
    p = ax.pcolormesh(lon, lat, variable_at_doy, **kwargs)
    plt.colorbar(p, ax=ax, shrink=0.6, label=varname)

    if cross is not None:
        plt.plot(cross[1], cross[0], "x", markersize=20, markeredgewidth=3, color="r")
    if circle is not None:
        plt.plot(
            circle[1],
            circle[0],
            "o",
            markersize=39,
            markeredgewidth=1,
            color="g",
            fillstyle="none",
        )

    # Label axes of a Plate Carree projection with a central longitude of 180:
    ax.set_global()
    ax.coastlines()
    ax.set_xticks([-180, -120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree())
    ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)

    plt.title(
        "regression "
        + coeff_name
        + " for "
        + varname
        + ", day of year: "
        + str(day_of_year)
    )


def plot_1d_doy(data, rstat, variable, lat_ind, lon_ind):

    """ Take a 1d slice with doy on x axis """

    #  lat = data.variables['lat'][lat_ind]
    #  lon = data.variables['lon'][lon_ind]
    #  fig = plt.plot(data_1d)
    fig = plt.figure()
    #  data_1d = data.variables[rstat][:, lat_ind, lon_ind]
    data_1d = data[:, lat_ind, lon_ind]
    plt.plot(data_1d)
    plt.xlabel("Day of Year")
    plt.ylabel(rstat)
    plt.title("var: " + variable)
    # plt.show()


def contourf_doy(data, doy, levels=30):
    # Take a 2d slice of slope for one doy and plot as filled contour
    data_2d = data[doy - 1, :, :]
    contour = qplt.contourf(data_2d, levels)
    plt.gca().coastlines()
    # plt.clabel(contour, inline=False)
    # plt.show()


def regr_fit_gmt(rdata, data, gmt, indices=(0, 0, 0), detrend=False):
    """ This functions fits, plots and saves a regression against global mean temperature"""

    length_of_year = rdata.variables["time"].shape[0]
    var = list(data.variables)[-1]
    gmt_var = list(gmt.variables)[0]

    doy = np.arange(length_of_year)
    lat = rdata.variables["lat"][indices[1]]
    lon = rdata.variables["lon"][indices[2]]
    slope = rdata.variables["slope"][indices]
    intercept = rdata.variables["intercept"][indices]

    plot_data = data.variables[var][
        indices[0] :: length_of_year, indices[1], indices[2]
    ]
    plot_gmt = gmt.variables[gmt_var][indices[0] :: length_of_year]
    fit = intercept + (slope * plot_gmt)

    time_values = np.arange(40150 / length_of_year) + 1901

    if detrend:
        plot_data = plot_data - fit + fit[0]
        s, i, r, p, sd = stats.linregress(plot_gmt, plot_data)
        fit = (s * plot_gmt) + i

    # Plot the data of one point and fit the regression.

    fig, ax = plt.subplots()
    ax.scatter(time_values, plot_data)
    ax.plot(time_values, fit, "r", label="fit")
    plt.title(
        "doy: " + str(doy[indices[0]]) + " lat: " + str(lat) + " lon: " + str(lon)
    )
    plt.xlabel("gmt / K")
    plt.ylabel(var + " / [unit]")
    plt.legend(ncol=2)
    plt.grid(True)
    plt.axis("tight")
    plt.show()
    # plt.savefig(os.path.join(set.data_dir, 'visual/') + var + '_gmt_doy' +
    #             str(doy[indices[0]]) + '_lat' + str(lat) + '_lon' + str(lon) + '.pdf',
    #            format='pdf')


def regr_fit_time(rdata, data, gmt, indices=(0, 0, 0), detrend=False):
    """ This functions fits, plots and saves a regression against time"""
    length_of_year = rdata.variables["time"].shape[0]
    var = list(data.variables)[-1]
    gmt_var = list(gmt.variables)[0]

    doy = np.arange(length_of_year)
    lat = rdata.variables["lat"][indices[1]]
    lon = rdata.variables["lon"][indices[2]]
    slope = rdata.variables["slope"][indices]
    intercept = rdata.variables["intercept"][indices]

    plot_data = data.variables[var][
        indices[0] :: length_of_year, indices[1], indices[2]
    ]
    plot_gmt = gmt.variables[gmt_var][indices[0] :: length_of_year]

    time_values = np.arange(40150 / 365) + 1901
    fit = intercept + (slope * plot_gmt)
    # fit = (fit * plot_gmt)

    if detrend:
        plot_data = plot_data - fit + fit[0]
        s, i, r, p, sd = stats.linregress(time_values, plot_data)
        fit = s * time_values + i

    # Plot the data of one point and fit the regression.
    fig, ax = plt.subplots()
    #  print(time_values.shape, plot_data.shape)
    ax.scatter(time_values, plot_data)
    ax.plot(time_values, fit, "r", label="fit")

    plt.title(
        "doy: " + str(doy[indices[0]]) + " lat: " + str(lat) + " lon: " + str(lon)
    )
    plt.xlabel("Year")
    plt.ylabel(var + " / [unit]")
    plt.legend(ncol=2)
    plt.grid(True)
    plt.axis("tight")
    plt.show()
    # plt.savefig(os.path.join(set.data_dir, 'visual/') + var + '_time_doy' +
    #             str(doy[indices[0]]) + '_lat' + str(lat) + '_lon' + str(lon) + '.pdf',
    #            format='pdf')


def regr_fit(rdata, data, gmt, indices=(0, 0, 0)):

    length_of_year = rdata.variables["time"].shape[0]
    var = list(data.variables)[-1]
    gmt_var = list(gmt.variables)[0]

    doy = np.arange(length_of_year)
    lat = rdata.variables["lat"][indices[1]]
    lon = rdata.variables["lon"][indices[2]]
    slope = rdata.variables["slope"][indices]
    intercept = rdata.variables["intercept"][indices]

    plot_data = data.variables[var][
        indices[0] :: length_of_year, indices[1], indices[2]
    ]
    plot_gmt = gmt.variables[gmt_var][indices[0] :: length_of_year]
    fit = intercept + (slope * plot_gmt)

    time_values = np.arange(40150 / length_of_year) + 1901

    plot_data_detrend = plot_data - fit + fit[0]
    s, i, r, p, sd = stats.linregress(plot_gmt, plot_data_detrend)
    fit_detrend = (s * plot_gmt) + i
    fit_time = s * time_values + i

    # Plot the data of one point and fit the regression.

    fig, ax = plt.subplots(2, 3, sharey="row", figsize=(16, 12))
    # fig.subplots_adjust(hspace=0, wspace=0.6)
    fig.suptitle(
        "var: "
        + var
        + " doy: "
        + str(doy[indices[0]])
        + " lat: "
        + str(lat)
        + " lon: "
        + str(lon),
        size=20,
        weight="bold",
    )

    ax[0, 0].scatter(plot_gmt, plot_data, label="data")
    ax[0, 0].plot(plot_gmt, fit, "r", label="fit")
    ax[0, 0].set_xlabel("gmt / K")
    ax[0, 0].set_ylabel(var + " / [unit]")
    ax[0, 0].grid()
    ax[0, 0].legend(ncol=1)

    ax[1, 0].scatter(plot_gmt, plot_data_detrend, label="detrended data")
    ax[1, 0].plot(plot_gmt, fit_detrend, "r", label="detrended fit")
    ax[1, 0].set_xlabel("gmt / K")
    ax[1, 0].set_ylabel(var + " / [unit]")
    ax[1, 0].grid()
    ax[1, 0].legend(ncol=1)

    ax[0, 1].scatter(time_values, plot_data, label="data")
    ax[0, 1].plot(time_values, fit, "r", label="fit")
    ax[0, 1].set_xlabel("gmt / K")
    ax[0, 1].set_ylabel(var + " / [unit]")
    ax[0, 1].grid()
    ax[0, 1].legend(ncol=1)

    ax[1, 1].scatter(time_values, plot_data_detrend, label="detrended data")
    ax[1, 1].plot(time_values, fit_time, "r", label="detrended fit")
    ax[1, 1].set_xlabel("gmt / K")
    ax[1, 1].set_ylabel(var + " / [unit]")
    ax[1, 1].grid()
    ax[1, 1].legend(ncol=1)

    ax[1, 2].plot(slope[lat, lon])
    ax[1, 2].set_xlabel("Day of Year")
    ax[1, 2].set_ylabel("slope")
    ax[1, 2].grid()

    ax[2, 2].plot(intercept[lat, lon])
    ax[2, 2].set_xlabel("Day of Year")
    ax[2, 2].set_ylabel("intercept")
    ax[2, 2].grid()

    plt.grid(True)
    plt.axis("tight")
    plt.show()
    # plt.savefig(os.path.join(set.data_dir, 'visual/') + var + '_doy' +
    #             str(doy[indices[0]]) + '_lat' + str(lat) + '_lon' + str(lon) + '.pdf',
    #            format='pdf')


def plot_2d_doy(data, doy, title):

    slope_2d = data.variables["slope"][doy, :, :]
    intercept_2d = data.variables["intercept"][doy, :, :]
    lat = data.variables["lat"][:]
    lon = data.variables["lon"][:]

    plt.figure(figsize=(16, 10))

    # Label axes of a Plate Carree projection with a central longitude of 180:
    ax = plt.subplot(121, projection=ccrs.PlateCarree(central_longitude=0))
    ax.set_global()
    ax.coastlines()
    ax.set_xticks([-180, -120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree())
    ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax2 = plt.pcolormesh(lon, lat, slope_2d, cmap="seismic")
    plt.title("var: " + title + " -- slope -- doy: " + str(doy))
    plt.colorbar(orientation="horizontal")

    ax = plt.subplot(122, projection=ccrs.PlateCarree(central_longitude=0))
    ax.set_global()
    ax.coastlines()
    ax.set_xticks([-180, -120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree())
    ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax2 = plt.pcolormesh(lon, lat, intercept_2d, cmap="coolwarm")
    plt.title("var: " + title + " -- intercept -- doy: " + str(doy))
    # plt.colorbar(orientation='horizontal')

    plt.show()
