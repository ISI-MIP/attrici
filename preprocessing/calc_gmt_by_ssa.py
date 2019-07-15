import sys
from datetime import datetime
import numpy as np
import iris
import iris.coord_categorisation as icc
from pyts.decomposition import SSA
import netCDF4 as nc

if "../" not in sys.path: sys.path.append("../")
import settings as s

# SSA options
# real window size will be window size* (selection step in col)
window_size = 365
grouping = 1

# specify paths
# script for standard output
out_script = '../output/gmt.out'
source_path = s.data_dir + "/input/tas_" + s.dataset + '_sub_gmt.nc4'
dest_path = s.data_dir + "/input/" + s.dataset + '_ssa_gmt.nc4'
print('Source path:')
print(source_path)
print('Destination path:')
print(dest_path)

# Get jobs starting time
STIME = datetime.now()
with open(out_script, 'w') as out:
    out.write('Job started at: ' + str(STIME) + '\n')
print('Job started at: ' + str(STIME))

# load data and add auxiliary variable "doy"
data = iris.load_cube(source_path, )
torigin = str(data.coord('time').units)
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

output_ds = nc.Dataset(dest_path, "w", format="NETCDF4")
time = output_ds.createDimension("time", None)
times = output_ds.createVariable("time", "f8", ("time",))
tas = output_ds.createVariable('tas', "f8", ("time"))

output_ds.description = "GMT created from daily values by SSA (10 day step)"
times.units = torigin
times.calendar = "365_day"

times[:] = range(0, 10*len(X_ssa[0, 0]), 10)
tas[:] = X_ssa[0, 0]
output_ds.close()
