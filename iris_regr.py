import os
import numpy as np
import iris
import iris.coord_categorisation as icc
from joblib import dump, load
import joblib
import multiprocessing
import psutil
from scipy import stats
import shutil
import sys
from datetime import datetime
import settings as s

print('Number of available CPUs:\n')
print('CPUs for multiprocessing:\n')
print (multiprocessing.cpu_count())
print('CPUs in total:\n')
print (os.cpu_count())
sys.stdout.flush()

#  specify paths

logfile = 'regr.out'
source_path_data = s.data_dir + 'tas_rm_rechunked_gswp3_1901_2010.nc4'
source_path_gmt = s.data_dir + 'gmt_gswp3_1901_2010.nc4'
dest_path = s.data_dir

#  Get jobs starting time
STIME = datetime.now()
with open(os.path.join(s.data_dir, logfile), 'w') as out:
    out.write('Job started at: ' + str(STIME) + '\n')
print('Job started at: ' + str(STIME))

#  load data
gmt = iris.load_cube(source_path_gmt)
icc.add_day_of_year(gmt, 'time')
print('GMT data is loaded and looks like this:\n')
print(gmt)
sys.stdout.flush()
data = iris.load_cube(source_path_data)
icc.add_day_of_year(data, 'time')
print('Target variable is loaded and looks like this:\n')
print(data)
sys.stdout.flush()

#  Get dayofyear-vectors of gmt and data
doys_cube = data.coord('day_of_year').points
#  print(doys_cube)
doys = gmt.coord('day_of_year').points
print('Days of Year are:\n')
print(doys)
sys.stdout.flush()

#  Create numpy arrays as containers for regression output
print('Creating Container for Slopes')
slope = np.ones((366,
                  data.coord('latitude').shape[0],
                  data.coord('longitude').shape[0]), dtype=np.float64)
intercept=slope
r_value=slope
p_value=slope
std_err=slope

print('Created Container for output data!')
print('Parent process stats:\n')
print('CPU % used:' , psutil.cpu_percent())
print('Memory % used:', psutil.virtual_memory()[2])
pid = os.getpid()
py = psutil.Process(pid)
memoryUse = py.memory_info()[0]/2.**30  # memory use in GB...I think
print('Memory use:', memoryUse)
sys.stdout.flush()


# loop over dayofyear-vector, then lon and lat and calculate regression
def regr_doy(doy):
    gmt_day = gmt[doys == doy].data
    #  A = np.vstack([gmt_day, np.ones(len(gmt_day))]).T
    #  write dayofyear to standard output
    for yx_slice in data.slices(['day_of_year']):
        sys.stdout.write('\nDayofYear is:\n' + str(doy) + '\n')
        sys.stdout.flush()
        s, i, r, p, sd = stats.linregress(gmt_day[1:-1],yx_slice[doys == doy].data[1:-1])
        #  sys.stdout.write('\nSlope is:\n')
        #  sys.stdout.write(str(s))
        #  print('\nIntercept is:\n')
        #  sys.stdout.write(str(i))
        lat = int(np.where(data.coord('latitude').points == yx_slice.coord('latitude').points)[0])
        sys.stdout.write('\nLatitude is:\n')
        sys.stdout.write(str(lat))
        lon = int(np.where(data.coord('longitude').points == yx_slice.coord('longitude').points)[0])
        sys.stdout.write('\nLongitude is:\n')
        sys.stdout.write(str(lon))
        sys.stdout.flush()
        #  write regression output to containers
        slope[doy-1, lat, lon] = s
        intercept[doy-1, lat, lon] = i
        r_value[doy-1, lat, lon] = r
        p_value[doy-1, lat, lon] = p
        std_err[doy-1, lat, lon] = sd

#  create memory map for child processes
folder = './joblib_memmap'
try:
    os.mkdir(folder)
except FileExistsError:
    pass

memmap = list()
for obj in ['slope', 'intercept', 'r_value', 'p_value', 'std_err']:
    memmap.append(os.path.join(folder, obj + '_memmap'))
    dump(eval(obj), memmap[-1])

slope = load(memmap[0], mmap_mode='r+')
intercept = load(memmap[1], mmap_mode='r+')
r_value = load(memmap[2], mmap_mode='r+')
p_value = load(memmap[3], mmap_mode='r+')
std_err = load(memmap[4], mmap_mode='r+')

#  run regression in parallel
print('Start with regression Calculations\n')
joblib.Parallel(n_jobs=-1, backend='multiprocessing')(
    joblib.delayed(regr_doy)(doy) for doy in np.unique(doys))

#  Clean up memory map
try:
    shutil.rmtree(folder)
except:  # noqa
    print('Could not clean-up automatically.')

#  Create dayofyear coordinate
doy_coord = iris.coords.DimCoord(range(1,367))
for obj in ['slope', 'intercept', 'r_value', 'p_value', 'std_err']:
    dest_path_obj = os.path.join(dest_path, 'test_' + data.name() + '_' + obj + '.nc4')
    #  wrap iris cube container around data
    cube = iris.cube.Cube(eval(obj),
                        dim_coords_and_dims=[(doy_coord, 0),
                                             (data.coord('latitude'), 1),
                                             (data.coord('longitude'), 2),
                                            ])
    iris.fileformats.netcdf.save(cube, dest_path_obj)

# Get jobs finishing time
FTIME = datetime.now()
with open(os.path.join(s.data_dir, logfile), 'a') as out:
    out.write('Job finished at: ' + str(FTIME) + '\n')
print('Job finished at: ' + str(FTIME))
duration = FTIME - STIME
print('Time elapsed ' +
      str(divmod(duration.total_seconds(), 3600)[0]) +
      ' hours!')
