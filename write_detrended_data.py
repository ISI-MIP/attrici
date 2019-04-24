import os
import sys
import importlib
import time as t
from datetime import datetime
import numpy as np
import netCDF4 as nc
import idetrend as idtr
import idetrend.const as c
import idetrend.visualization as vis

importlib.reload(vis)
import settings as s

# parameters and data paths

# the file with the smoothed global trend of global mean temperature
gmt_file = os.path.join(s.data_dir, s.gmt_file)
# the daily interpolated ssa-smoothed global mean temperature
gmt_on_each_day = idtr.utility.get_gmt_on_each_day(gmt_file, s.days_of_year)


varname = s.variable
if s.test:
    file_to_write = os.path.join(s.data_dir, varname + "_detrended_test.nc4")
else:
    file_to_write = os.path.join(s.data_dir, varname + "_detrended.nc4")

regression_file = os.path.join(s.data_dir, s.regression_outfile)
to_detrend_file = os.path.join(s.data_dir, s.to_detrend_file)


lats, lons, slope, intercept = vis.get_coefficient_fields(regression_file)

if s.test:
    shape_of_input = (40150, 12, 24)
else:
    shape_of_input = (40150, 360, 720)

if __name__ == "__main__":

    TIME0 = datetime.now()

    idtr.lin_regr.write_detrended(
        regression_file,
        to_detrend_file,
        shape_of_input,
        (lats, lons),
        file_to_write,
        varname,
        gmt_on_each_day
    )

    TIME1 = datetime.now()
    duration = TIME1 - TIME0
    print("Calculation took", duration.total_seconds(), "seconds.")
