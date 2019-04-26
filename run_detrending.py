import os
import sys
import importlib
import time as t
from datetime import datetime
import numpy as np
import netCDF4 as nc
import idetrend as idtr
import idetrend.const as c

import settings as s

# the file with the smoothed global trend of global mean temperature
gmt_file = os.path.join(s.data_dir, s.gmt_file)
# the daily interpolated ssa-smoothed global mean temperature
gmt_on_each_day = idtr.utility.get_gmt_on_each_day(gmt_file, s.days_of_year)

regression_file = os.path.join(s.data_dir, s.regression_outfile)
to_detrend_file = os.path.join(s.data_dir, s.to_detrend_file)
detrended_file = os.path.join(s.data_dir, s.detrended_file)

lats, lons, slope, intercept = idtr.detrending.get_coefficient_fields(regression_file)

if __name__ == "__main__":

    TIME0 = datetime.now()

    detrend = idtr.detrending.detrending(
        lons,
        lats,
        slope,
        intercept,
        to_detrend_file,
        s.variable,
        gmt_on_each_day,
        s.days_of_year,
    )

    detrend.write_detrended(detrended_file)

    TIME1 = datetime.now()
    duration = TIME1 - TIME0
    print("Calculation took", duration.total_seconds(), "seconds.")
