import datetime
import iris
import iris.coord_categorisation as icc
import numpy as np

# specify paths
out_script = '/home/bschmidt/scripts/detrending/ouput/gmt.out'
source_path = '/home/bschmidt/data/test_data_tas.nc4'
dest_path = '/home/bschmidt/data/test_gmt.nc'

# Get jobs starting time
STIME = datetime.now()
with open(out_script, 'w') as out:
    out.write('Job started at: ' + str(STIME) + '\n')
print('Job started at: ' + str(STIME))

# load data and add auxiliary variable "doy"
data = iris.load_cube(source_path)
icc.add_day_of_year(data, 'time')

#  Let iris figure out cell boundaries and calculate
#  global mean temperatures weighted by cell area
data.coord('latitude').guess_bounds()
data.coord('longitude').guess_bounds()
grid_areas = iris.analysis.cartography.area_weights(data)
# .collapsed method applieas a function to a grid and outputs
col = data.collapsed(['longitude', 'latitude'],
                                iris.analysis.MEAN,
                                weights=grid_areas)

# pad time coordinate by half window size (182 for yearly averaging)
# TODO: Padding of time coordinate ensures that dimension
#       is not cut off, but ends are only 'pseudo-averaged'
#       This results in outliers, that have great influence on slopes
#       What do we do?
gmt = np.pad(col.data, 182, 'edge')
# apply rolling window operator and run average on it
gmt = np.mean(iris.util.rolling_window(gmt, window=365, step=1), -1)
# create iris cube as container around numpy data (gmt)
# with coordinates taken from input data cube (tas)
gmt = iris.cube.Cube(gmt,
                     var_name=col.var_name,
                     cell_methods=col.cell_methods,
                     dim_coords_and_dims=[(col.coord('time'), 0)],
                     aux_coords_and_dims=[(col.coord('day_of_year'), 0)])

# save cube to netCDF4 file
iris.save(gmt, dest_path, netcdf_format="NETCDF4")

# Get jobs finishing time
FTIME = datetime.now()
with open(out_script, 'a') as out:
    out.write('Job finished at: ' + str(FTIME) + '\n')
print('Job finished at: ' + str(FTIME))
duration = FTIME - STIME
print('Time elapsed ' +
      str(divmod(duration.total_seconds(), 3600)[0]) +
      ' hours!')
