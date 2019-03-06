import os
#  import time
#  from multiprocess import Pool
import numpy as np
import iris
import iris.coord_categorisation as icc
#  from scipy import stats
#  import dask as da
from datetime import datetime


#  specify paths
out_script = '/home/bschmidt/scripts/detrending/output/regr.out'
source_path_data = '/home/bschmidt/data/test_data_tas.nc4'
source_path_gmt = '/home/bschmidt/data/test_gmt.nc4'
dest_path_intercept = '/home/bschmidt/data/test_tas_intercept.nc'
dest_path_slope = '/home/bschmidt/data/test_tas_slope.nc'

#  Get jobs starting time
STIME = datetime.now()
with open(out_script, 'w') as out:
    out.write('Job started at: ' + str(STIME) + '\n')
print('Job started at: ' + str(STIME))

#  load data
gmt = iris.load_cube(source_path_gmt)
#  icc.add_day_of_year(gmt, 'time')
#  print(gmt)
data = iris.load_cube(source_path_data)
icc.add_day_of_year(data, 'time')
#  print(data)

#  Get dayofyear-vectors of gmt and data
doys_cube = data.coord('day_of_year').points
#  print(doys_cube)
doys = gmt.coord('day_of_year').points
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
def regr_doy(doy):
    gmt_day = gmt[doys == doy].data
    A = np.vstack([gmt_day, np.ones(len(gmt_day))]).T
    print('\nDayofYear is:\n' + str(doy))
    for yx_slice in data.slices(['day_of_year']):
        slope, intercept = np.linalg.lstsq(A, yx_slice[doys == doy].data)[0]
        #print('\nSlope is:\n')
        #print(slope)
        #print('\nIntercept is:\n')
        #print(intercept)
        #print('\nDayofYear is:\n')
        #print(doy)
        lat = int(np.where(data.coord('latitude').points == yx_slice.coord('latitude').points)[0])
        #print('\nLatitude is:\n')
        #print(lat)
        lon = int(np.where(data.coord('longitude').points == yx_slice.coord('longitude').points)[0])
        #print('\nLongitude is:\n')
        #print(lon)
        #  write regression output to containers
        slopes[doy-1, lat, lon] = slope
        intercepts[doy-1, lat, lon] = intercept

#  create memory map for child processes
folder = './joblib_memmap'
try:
    os.mkdir(folder)
except FileExistsError:
    pass

slopes_memmap = os.path.join(folder, 'slopes_memmap')
dump(slopes, slopes_memmap)
slopes = load(slopes_memmap, mmap_mode='r+')

intercepts_memmap = os.path.join(folder, 'intercepts_memmap')
dump(intercepts, intercepts_memmap)
intercepts = load(intercepts_memmap, mmap_mode='r+')

#  run regression in parallel
print('Start with regression Calculations\n')
joblib.Parallel(n_jobs=5)(
    joblib.delayed(regr_doy)(doy) for doy in np.unique(doys))

#  Clean up memory map
try:
    shutil.rmtree(folder)
except:  # noqa
    print('Could not clean-up automatically.')

#  Create dayofyear coordinate
doy_coord = iris.coords.DimCoord(range(1,367))
#  wrap iris cube container around data
slopes = iris.cube.Cube(slopes,
                        dim_coords_and_dims=[(doy_coord, 0),
                                             (data.coord('latitude'), 1),
                                             (data.coord('longitude'), 2),
                                            ])
#  save slope data to netCDF4
iris.fileformats.netcdf.save(slopes, dest_path_slope)

#  repeat saving for intercept
intercepts = iris.cube.Cube(intercepts,
                        dim_coords_and_dims=[(doy_coord, 0),
                                             (data.coord('latitude'), 1),
                                             (data.coord('longitude'), 2),
                                            ])
iris.fileformats.netcdf.save(intercepts, dest_path_intercept)

# Get jobs finishing time
FTIME = datetime.now()
with open(out_script, 'a') as out:
    out.write('Job finished at: ' + str(FTIME) + '\n')
print('Job finished at: ' + str(FTIME))
duration = FTIME - STIME
print('Time elapsed ' +
      str(divmod(duration.total_seconds(), 3600)[0]) +
      ' hours!')
