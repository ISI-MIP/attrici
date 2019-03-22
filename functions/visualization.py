#!/usr/bin/env python3
# coding: utf-8

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa
from scipy import stats
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import os
import settings as set

def plot_1d_doy(data, rstat, variable, lat_ind, lon_ind):

    ''' Take a 1d slice with doy on x axis '''

    lat = data.variables['lat'][lat_ind]
    lon = data.variables['lon'][lon_ind]
    data_1d = data.variables[rstat][:, lat_ind, lon_ind]
    # fig = plt.plot(data_1d)
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    for i in range(1, 2):
        ax = fig.add_subplot(2, 1, i)
        ax.plot(data_1d)
        plt.xlabel('Day of Year')
        plt.ylabel(rstat)
    plt.title('var: ' + variable + ' lat: ' + str(lat) + ' lon: ' + str(lon))
    # plt.show()


def contourf_doy(data, doy, levels=30):
    # Take a 2d slice of slope for one doy and plot as filled contour
    data_2d = data[doy-1, :, :]
    contour = qplt.contourf(data_2d, levels)
    plt.gca().coastlines()
    #plt.clabel(contour, inline=False)
    # plt.show()


def regr_fit_gmt(rdata, data, gmt, indices=(0, 0, 0), detrend=False):
    ''' This functions fits, plots and saves a regression against global mean temperature'''

    length_of_year = rdata.variables['time'].shape[0]
    var = list(data.variables)[-1]
    gmt_var = list(gmt.variables)[0]

    doy = np.arange(length_of_year)
    lat = rdata.variables['lat'][indices[1]]
    lon = rdata.variables['lon'][indices[2]]
    slope = rdata.variables['slope'][indices]
    intercept = rdata.variables['intercept'][indices]

    plot_data = data.variables[var][indices[0]::length_of_year, indices[1], indices[2]]
    plot_gmt = gmt.variables[gmt_var][indices[0]::length_of_year]
    fit = intercept + (slope * plot_gmt)

    time_values = np.arange(40150/length_of_year) + 1901

    if detrend:
        plot_data = plot_data - fit + fit[0]
        s, i, r, p, sd = stats.linregress(plot_gmt, plot_data)
        fit = (s * plot_gmt) + i

    # Plot the data of one point and fit the regression.

    fig, ax = plt.subplots()
    ax.scatter(time_values, plot_data)
    ax.plot(time_values, fit, 'r', label='fit')
    plt.title('doy: ' + str(doy[indices[0]]) + ' lat: ' + str(lat) + ' lon: ' + str(lon))
    plt.xlabel('gmt / K')
    plt.ylabel(var + ' / [unit]')
    plt.legend(ncol=2)
    plt.grid(True)
    plt.axis('tight')
    plt.show()
    # plt.savefig(os.path.join(set.data_dir, 'visual/') + var + '_gmt_doy' +
    #             str(doy[indices[0]]) + '_lat' + str(lat) + '_lon' + str(lon) + '.pdf',
    #            format='pdf')


def regr_fit_time(rdata, data, gmt, indices=(0, 0, 0), detrend = False):
    ''' This functions fits, plots and saves a regression against time'''
    length_of_year = rdata.variables['time'].shape[0]
    var = list(data.variables)[-1]
    gmt_var = list(gmt.variables)[0]

    doy = np.arange(length_of_year)
    lat = rdata.variables['lat'][indices[1]]
    lon = rdata.variables['lon'][indices[2]]
    slope = rdata.variables['slope'][indices]
    intercept = rdata.variables['intercept'][indices]

    plot_data = data.variables[var][indices[0]::length_of_year, indices[1], indices[2]]
    plot_gmt = gmt.variables[gmt_var][indices[0]::length_of_year]

    time_values = np.arange(40150/365) + 1901
    fit = intercept + (slope * plot_gmt)
    #fit = (fit * plot_gmt)

    if detrend:
        plot_data = plot_data - fit + fit[0]
        s, i, r, p, sd = stats.linregress(time_values, plot_data)
        fit = s * time_values + i

    # Plot the data of one point and fit the regression.
    fig, ax = plt.subplots()
    #  print(time_values.shape, plot_data.shape)
    ax.scatter(time_values, plot_data)
    ax.plot(time_values, fit, 'r', label='fit')

    plt.title('doy: ' + str(doy[indices[0]]) + ' lat: ' + str(lat) + ' lon: ' + str(lon))
    plt.xlabel('Year')
    plt.ylabel(var + ' / [unit]')
    plt.legend(ncol=2)
    plt.grid(True)
    plt.axis('tight')
    plt.show()
    # plt.savefig(os.path.join(set.data_dir, 'visual/') + var + '_time_doy' +
    #             str(doy[indices[0]]) + '_lat' + str(lat) + '_lon' + str(lon) + '.pdf',
    #            format='pdf')


def regr_fit(rdata, data, gmt, indices=(0, 0, 0)):

    length_of_year = rdata.variables['time'].shape[0]
    var = list(data.variables)[-1]
    gmt_var = list(gmt.variables)[0]

    doy = np.arange(length_of_year)
    lat = rdata.variables['lat'][indices[1]]
    lon = rdata.variables['lon'][indices[2]]
    slope = rdata.variables['slope'][indices]
    intercept = rdata.variables['intercept'][indices]

    plot_data = data.variables[var][indices[0]::length_of_year, indices[1], indices[2]]
    plot_gmt = gmt.variables[gmt_var][indices[0]::length_of_year]
    fit = intercept + (slope * plot_gmt)

    time_values = np.arange(40150/length_of_year) + 1901

    plot_data_detrend = plot_data - fit + fit[0]
    s, i, r, p, sd = stats.linregress(plot_gmt, plot_data_detrend)
    fit_detrend = (s * plot_gmt) + i
    fit_time = s * time_values + i

    # Plot the data of one point and fit the regression.

    fig, ax = plt.subplots(2, 2, sharey='row', figsize=(16, 12))
    #fig.subplots_adjust(hspace=0, wspace=0.6)
    fig.suptitle('var: ' + var + ' doy: ' + str(doy[indices[0]]) +
                 ' lat: ' + str(lat) + ' lon: ' + str(lon), size=20, weight='bold')

    ax[0, 0].scatter(plot_gmt, plot_data, label='data')
    ax[0, 0].plot(plot_gmt, fit, 'r', label='fit')
    ax[0, 0].set_xlabel('gmt / K')
    ax[0, 0].set_ylabel(var + ' / [unit]')
    ax[0, 0].grid()
    ax[0, 0].legend(ncol=1)

    ax[1, 0].scatter(plot_gmt, plot_data_detrend, label='detrended data')
    ax[1, 0].plot(plot_gmt, fit_detrend, 'r', label='detrended fit')
    ax[1, 0].set_xlabel('gmt / K')
    ax[1, 0].set_ylabel(var + ' / [unit]')
    ax[1, 0].grid()
    ax[1, 0].legend(ncol=1)

    ax[0, 1].scatter(time_values, plot_data, label='data')
    ax[0, 1].plot(time_values, fit, 'r', label='fit')
    ax[0, 1].set_xlabel('gmt / K')
    ax[0, 1].set_ylabel(var + ' / [unit]')
    ax[0, 1].grid()
    ax[0, 1].legend(ncol=1)

    ax[1, 1].scatter(time_values, plot_data_detrend, label='detrended data')
    ax[1, 1].plot(time_values, fit_time, 'r', label='detrended fit')
    ax[1, 1].set_xlabel('gmt / K')
    ax[1, 1].set_ylabel(var + ' / [unit]')
    ax[1, 1].grid()
    ax[1, 1].legend(ncol=1)

    plt.grid(True)
    plt.axis('tight')
    plt.show()
    # plt.savefig(os.path.join(set.data_dir, 'visual/') + var + '_doy' +
    #             str(doy[indices[0]]) + '_lat' + str(lat) + '_lon' + str(lon) + '.pdf',
    #            format='pdf')

def plot_2d_doy(data, doy, title):

    slope_2d = data.variables['slope'][doy, :, :]
    intercept_2d = data.variables['intercept'][doy, :, :]
    lat = data.variables['lat'][:]
    lon = data.variables['lon'][:]

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
    ax2 = plt.pcolormesh(lon, lat, slope_2d, cmap='seismic')
    plt.title('var: ' + title + ' -- slope -- doy: ' + str(doy))
    plt.colorbar(orientation='horizontal')

    ax = plt.subplot(122, projection=ccrs.PlateCarree(central_longitude=0))
    ax.set_global()
    ax.coastlines()
    ax.set_xticks([-180, -120, -60, 0, 60, 120, 180], crs=ccrs.PlateCarree())
    ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax2 = plt.pcolormesh(lon, lat, intercept_2d, cmap='coolwarm')
    plt.title('var: ' + title + ' -- intercept -- doy: ' + str(doy))
    # plt.colorbar(orientation='horizontal')

    plt.show()
