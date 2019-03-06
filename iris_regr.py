import os
import time
from multiprocess import Pool
import sys
import numpy as np
import iris
import iris.coord_categorisation as icc
from scipy import stats
import dask as da
from datetime import datetime
os.chdir('/home/bschmidt/data/')

start_time = datetime.now()
gmt = iris.load_cube('test_gmt.nc4')
#  icc.add_day_of_year(gmt, 'time')
print(gmt)
duration_run = datetime.now() - start_time
print("\n GMT calculations' duration in seconds %4.2f \n" % (duration_run.total_seconds()))

start_time = datetime.now()
data = iris.load_cube('test_data_tas.nc4')
icc.add_day_of_year(data, 'time')
#print(data)

doys_cube = data.coord('day_of_year').points
#print(doys_cube)
doys = gmt.coord('day_of_year').points
#print(doys)
print('Creating Container for Slopes')
slopes = np.ones((366,
                  data.coord('latitude').shape[0],
                  data.coord('longitude').shape[0]), dtype=np.float64)
print('Creating Container for Intercepts')
intercepts = np.ones((366,
                      data.coord('latitude').shape[0],
                      data.coord('longitude').shape[0]), dtype=np.float64)
doys_cube = gmt.coord('day_of_year').points
print('Days of Year are:\n')
print(np.unique(doys))

print('Start with regression Calculations\n')
for doy in np.unique(doys):
    gmt_day = gmt[doys == doy].data
    A = np.vstack([gmt_day, np.ones(len(gmt_day))]).T
    for yx_slice in data.slices(['day_of_year']):
        slope, intercept = np.linalg.lstsq(A, yx_slice[doys == doy].data)[0]
        print('\nSlope is:\n')
        print(slope)
        print('\nIntercept is:\n')
        print(intercept)
        print('\nDayofYear is:\n')
        print(doy)
        lat = int(np.where(data.coord('latitude').points == yx_slice.coord('latitude').points)[0])
        print('\nLatitude is:\n')
        print(lat)
        lon = int(np.where(data.coord('longitude').points == yx_slice.coord('longitude').points)[0])
        print('\nLongitude is:\n')
        print(lon)
        slopes[doy-1, lat, lon] = slope
        intercepts[doy-1, lat, lon] = intercept


#  def regr_pool():
#      pool = Pool(processes=5)
#      chunks = [np.unique(doys)[i::5] for i in range(366)]
#      result = pool.map_async(regr, chunks)
#
#      while not result.ready():
#          print("Running...")
#          time.sleep(3)
#      return sum(result.get())
#
#  slopes, intercepts = regr_pool()

doy_coord = iris.coords.DimCoord(range(1,367))
slopes = iris.cube.Cube(slopes,
                        dim_coords_and_dims=[(doy_coord, 0),
                                             (data.coord('latitude'), 1),
                                             (data.coord('longitude'), 2),
                                            ])

iris.fileformats.netcdf.save(slopes, "test_tas_slope.nc4")
intercepts = iris.cube.Cube(intercepts,
                        dim_coords_and_dims=[(doy_coord, 0),
                                             (data.coord('latitude'), 1),
                                             (data.coord('longitude'), 2),
                                            ])
iris.fileformats.netcdf.save(intercepts, "test_tas_intercept.nc4", netcdf_format="NETCDF4")
