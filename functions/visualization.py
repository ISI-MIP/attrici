#!/usr/bin/env python3
# coding: utf-8

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa
import iris
import matplotlib.pyplot as plt
import iris.plot as iplt
import iris.quickplot as qplt
import iris.analysis.cartography
import iris.coord_categorisation as icc
import numpy as np

def plot_1d_doy(data, lat_ind, lon_ind):
    # Take a 1d slice with doy on x axis
    lat = data.coord('latitude').points[lat_ind]
    lon = data.coord('longitude').points[lon_ind]
    data_1d = data[:, lat_ind, lon_ind]
    fig = iplt.plot(data_1d)
    plt.title('Lat: ' + str(lat) + ' Lon: ' + str(lon))
    plt.xlabel('Day of Year')
    plt.ylabel(data.name())
    plt.show()


def contourf_doy(data, doy, levels=30):
    # Take a 2d slice of slope for one doy and plot as filled contour
    data_2d = data[doy-1, :, :]
    contour = qplt.contourf(data_2d, levels)
    plt.gca().coastlines()
    #plt.clabel(contour, inline=False)
    plt.show()


def regr_fit_gmt(slope, intercept, data, gmt, indices=(0, 0, 0)):
    doy = slope.coord('day_of_year').points[indices[0]]
    lat = slope.coord('latitude').points[indices[1]]
    lon = slope.coord('longitude').points[indices[2]]
    slope = slope[indices]
    intercept = intercept[indices]
    plot_data = data.extract(iris.Constraint(day_of_year=indices[0]+1))
    plot_data = plot_data[:, indices[1], indices[2]]
    fit = intercept.data + (slope.data * gmt.data)
    # Plot the data of one point and fit the regression.
    plt.scatter(gmt.data, plot_data.data)
    plt.plot(gmt.data, fit, 'r', label='fit')
    plt.title('Doy: ' + str(doy) + ' Lat: ' + str(lat) + ' Lon: ' + str(lon))
    plt.xlabel('gmt / K')
    plt.ylabel(data.name() + ' / K')
    plt.legend(ncol=2)
    plt.grid(True)
    plt.axis('tight')
    iplt.show()


def regr_fit_time(slope, intercept, data, gmt):
    slope = slope[1, 6, 12]
    intercept = intercept[1, 6, 12]
    plot_data = data.extract(iris.Constraint(day_of_year=1))
    plot_data = plot_data[:, 6, 12]
    fit = intercept.data + (slope.data * gmt.data)
    time_values = np.round(gmt.coord('time').points/365 + 1901)
    # Plot the data of one point and fit the regression.
    fig, ax = plt.subplots()
    ax.scatter(time_values, plot_data.data)
    ax.plot(time_values, fit, 'r', label='fit')
    plt.xlabel('Year')
    plt.ylabel('tas / K')
    ax.xaxis.set_major_locator(plt.MaxNLocator(10))
    plt.legend(ncol=2)
    plt.grid(True)
    plt.axis('tight')
    iplt.show()


# Application of visualization functions

# Data paths
slope_path = '/home/bschmidt/data/test_tas_slope.nc4'
intercept_path = '/home/bschmidt/data/test_tas_intercept.nc4'
gmt_path = '/home/bschmidt/data/test_gmt.nc4'
tas_path = '/home/bschmidt/data/test_data_tas.nc4'

# Load data and rename variables and coordinates
slope = iris.load_cube(slope_path)
slope.rename('slope')
slope.coord('dim0').rename('day_of_year')
intercept = iris.load_cube(intercept_path)
intercept.rename('intercept')
intercept.coord('dim0').rename('day_of_year')
gmt = iris.load_cube(gmt_path)
tas = iris.load_cube(tas_path)
icc.add_day_of_year(tas, 'time')

# create plots
plot_1d_doy(slope, lat_ind=0, lon_ind=0)
plot_1d_doy(intercept,lat_ind=0, lon_ind=0)
contourf_doy(slope, doy=1)
contourf_doy(intercept, doy=1)
regr_fit_gmt(slope=slope, intercept=intercept, data=tas, gmt=gmt)
regr_fit_time(slope=slope, intercept=intercept, data=tas, gmt=gmt)


