#!/usr/bin/env python
# coding: utf-8

import os
import sys
import numpy as np
import netCDF4 as nc
from scipy import special
from datetime import datetime, timedelta
import time as t
from operator import itemgetter

# import regression
import settings as s
import idetrend as idtr

gmt_file = os.path.join(s.data_dir, s.gmt_file)
to_detrend_file = os.path.join(s.data_dir, s.to_detrend_file)

gmt_on_each_day = idtr.utility.get_gmt_on_each_day(gmt_file, s.days_of_year)
data = nc.Dataset(to_detrend_file, "r")

# FIXME: such code needs to be avoided. Why not explicitely using
# the direct name from settings anyway?
# var = list(data.variables.keys())[-1]

data_to_detrend = data.variables[s.variable]
data_to_detrend = idtr.utility.check_data(data_to_detrend, to_detrend_file)
#  data_to_detrend = special.logit(data/100)

if __name__ == "__main__":

    TIME0 = datetime.now()

    regr = idtr.lin_regr.regression(gmt_on_each_day)
    results = idtr.utility.run_function_on_ncdf(
        data_to_detrend,
        # gmt_on_each_day,
        s.days_of_year,
        regr,
        s.n_jobs,
    )

    # results = run_parallel_linear_regr(n_jobs=3)
    TIME1 = datetime.now()
    duration = TIME1 - TIME0
    print("Calculation took", duration.total_seconds(), "seconds.")

    file_to_write = os.path.join(s.data_dir, s.regression_outfile)
    # due to a bug in iris I guess, I cannot overwrite existing files. Remove before.
    if os.path.exists(file_to_write):
        os.remove(file_to_write)
    idtr.lin_regr.write_regression_stats(
        data_to_detrend.shape,
        (data.variables["lat"], data.variables["lon"]),
        results,
        file_to_write,
        s.days_of_year,
    )
    TIME2 = datetime.now()
    duration = TIME2 - TIME1
    print("Saving took", duration.total_seconds(), "seconds.")
