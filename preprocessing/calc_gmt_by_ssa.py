from datetime import datetime
import numpy as np
import iris
import iris.coord_categorisation as icc
from pyts.decomposition import SSA
import netCDF4 as nc

# SSA options
# real window size will be window size* (selection step in col)
window_size = 365
grouping = 1

# specify paths
out_script = '/home/bschmidt/scripts/detrending/output/gmt.out'
source_path = '/home/bschmidt/temp/gswp3/test_data_tas.nc4'
dest_path = '/home/bschmidt/temp/gswp3/test_gmt.nc'

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
del data

# select every 10th timestep
col = np.array(col.data[::10], ndmin=2)

ssa = SSA(window_size)
X_ssa = ssa.fit_transform(col)


output_ds = nc.Dataset('/home/bschmidt/temp/gswp3/test_ssa_gmt.nc4', "w", format="NETCDF4")
time = output_ds.createDimension("time", None)
times = output_ds.createVariable("time", "f8", ("time",))
tas = output_ds.createVariable('tas', "f8", ("time"))

output_ds.description = "GMT created from daily values by SSA (10 day step)"
times.units = "days since 1901-01-01 00:00:00.0"
times.calendar = "365_day"

times[:] = range(0, 40150, 10)
tas[:] = X_ssa[0, 0]
output_ds.close()
