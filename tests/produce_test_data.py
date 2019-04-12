""" we here take subsets of our netcdf data and save it to plain csv files.
    These are stored in the git repository, so that code can be tested with
    this data without having netcdf files ready.
    This file does not need to be run regularly under pytest.
    Just use it if you want to update the test data. """

import sys
import os
import netCDF4 as nc
import pandas as pd

if ".." not in sys.path:
    sys.path.append("..")
import settings as s
import idetrend as idtr

gmt_file = os.path.join(s.data_dir, s.gmt_file)
to_detrend_file = os.path.join(s.data_dir, s.to_detrend_file)

gmt_on_each_day = idtr.utility.get_gmt_on_each_day(gmt_file, s.days_of_year)
data = nc.Dataset(to_detrend_file, "r")

data_to_detrend = data.variables["tas"]
pd.DataFrame(data_to_detrend[:, 3, 5]).to_csv("data/tas_testdata.csv", header=None)
