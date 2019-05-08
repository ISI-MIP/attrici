#!/usr/bin/env python
# coding: utf-8

import os
#  import sys
#  import numpy as np
import netCDF4 as nc
#  from scipy import special
from datetime import datetime
#  import time as t
#  from operator import itemgetter
import settings as s
import idetrend as idtr
import idetrend.const as c

gmt_file = os.path.join(s.data_dir, s.gmt_file)
to_detrend_file = os.path.join(s.data_dir, s.to_detrend_file)

gmt_on_each_day = idtr.utility.get_gmt_on_each_day(gmt_file, s.days_of_year)
data = nc.Dataset(to_detrend_file, "r")

data_to_detrend = data.variables[s.variable]  # [:]
data_to_detrend = idtr.utility.check_data(data_to_detrend, to_detrend_file)

# data_to_detrend = idtr.utility.mask_invalid(
#     data_to_detrend, idtr.const.minval[s.variable], idtr.const.maxval[s.variable]
# )

if __name__ == "__main__":

    TIME0 = datetime.now()
    print("Variable is:")
    print(s.variable, flush=True)

    regr = idtr.regression.regression(
        gmt_on_each_day,
        s.min_ts_len,
        c.minval[s.variable],
        c.maxval[s.variable],
        c.transform[s.variable],
    )

    results = idtr.utility.run_regression_on_dataset(
        data_to_detrend, s.days_of_year, regr, s.n_jobs
    )

    TIME1 = datetime.now()
    duration = TIME1 - TIME0
    print("Calculation took", duration.total_seconds(), "seconds.")

    file_to_write = os.path.join(s.data_dir, s.regression_outfile)

    if os.path.exists(file_to_write):
        os.remove(file_to_write)

    idtr.regression.write_regression_stats(
        data_to_detrend.shape,
        (data.variables["lat"], data.variables["lon"]),
        results,
        file_to_write,
        s.days_of_year,
    )
    TIME2 = datetime.now()
    duration = TIME2 - TIME1
    print("Saving took", duration.total_seconds(), "seconds.")
