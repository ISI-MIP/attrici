import os
#  import time
#  from multiprocess import Pool
import numpy as np
import iris
import iris.coord_categorisation as icc
#  from scipy import stats
#  import dask as da
from datetime import datetime
import settings as s


#  specify paths
logfile = 'regr.out'
test_data = 'test_data_tas.nc4'
test_gmt = 'test_gmt.nc4'
intercept_outfile = 'test_tas_intercept.nc'
slope_outfile = 'test_tas_slope.nc'

#  Get jobs starting time
STIME = datetime.now()
with open(os.path.join(s.data_dir,logfile), 'w') as out:
    out.write('Job started at: ' + str(STIME) + '\n')
print('Job started at: ' + str(STIME))

#  load data
gmt = iris.load_cube(os.path.join(s.data_dir,test_gmt))
#  icc.add_day_of_year(gmt, 'time')
#  print(gmt)
data = iris.load_cube(os.path.join(s.data_dir,test_data))
icc.add_day_of_year(data, 'time')
#  print(data)

#  Get dayofyear-vectors of gmt and data
doys_cube = data.coord('day_of_year').points
#  print(doys_cube)
doys = np.unique(gmt.coord('day_of_year').points)
print('Days of Year are:\n')
print(doys)

#  Create numpy arrays as containers for regression output
print('Creating Container for Slopes')
slopes = np.ones((366,
                  data.coord('latitude').shape[0],
                  data.coord('longitude').shape[0]), dtype=np.float64)
print('Creating Container for Intercepts')
intercepts = np.ones((366,
                      data.coord('latitude').shape[0],
                      data.coord('longitude').shape[0]), dtype=np.float64)


# loop over dayofyear-vector, then lon and lat and calculate regression
# TODO: run this in parallel
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
        #  write regression output to containers
        slopes[doy-1, lat, lon] = slope
        intercepts[doy-1, lat, lon] = intercept

#  FIRST TRY AT MULTIPROCESSING
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

#  Create dayofyear coordinate
doy_coord = iris.coords.DimCoord(range(1,367))
#  wrap iris cube container around data
slopes = iris.cube.Cube(slopes,
                        dim_coords_and_dims=[(doy_coord, 0),
                                             (data.coord('latitude'), 1),
                                             (data.coord('longitude'), 2),
                                            ])
#  save slope data to netCDF4
iris.fileformats.netcdf.save(slopes, os.path.join(s.data_dir,slope_outfile))

#  repeat saving for intercept
intercepts = iris.cube.Cube(intercepts,
                        dim_coords_and_dims=[(doy_coord, 0),
                                             (data.coord('latitude'), 1),
                                             (data.coord('longitude'), 2),
                                            ])
iris.fileformats.netcdf.save(intercepts, os.path.join(s.data_dir,intercept_outfile))

# Get jobs finishing time
FTIME = datetime.now()
with open(os.path.join(s.data_dir,logfile), 'a') as out:
    out.write('Job finished at: ' + str(FTIME) + '\n')
print('Job finished at: ' + str(FTIME))
duration = FTIME - STIME
print('Time elapsed ' +
      str(divmod(duration.total_seconds(), 3600)[0]) +
      ' hours!')
