import iris
import numpy as np
import iris.coord_categorisation as icc

# load data and add auxiliary variable "doy"
data_path = '/home/bschmidt/data/test_data_tas.nc4'

data = iris.load_cube(data_path)
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
iris.save(gmt, "test_gmt.nc", netcdf_format="NETCDF4")
