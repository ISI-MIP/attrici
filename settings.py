import getpass
import numpy as np
import idetrend.utility as u

user = getpass.getuser()

# this will hopefully avoid hand editing paths everytime.
# fill further for convenience.
if user == "mengel":
    data_dir = "/home/mengel/data/20190306_IsimipDetrend/"
elif user == "bschmidt":
    data_dir = "/home/bschmidt/temp/gswp3/"

#  handle job specifications
n_jobs = 16  # number of childprocesses created by the job

# if test=True use smaller test dataset
test = False
variable = "rhs"
dataset = "gswp3"
startyear = 1901
endyear = 2010

days_of_year = 365

gmt_file = dataset + "_ssa_gmt.nc4"
base_file = (
    variable
    + "_"
    + dataset
    + "_"
    + str(startyear)
    + "_"
    + str(endyear)
    + "_rechunked"
    + "_noleap.nc4"
)

if test:
    to_detrend_file = "test_data_" + variable + ".nc4"
    regression_outfile = variable + dataset + "_regression_test.nc4"
    detrended_file = variable + dataset + "_detrended_test.nc4"
else:
    to_detrend_file = (
        variable
        + "_"
        + dataset
        + "_"
        + str(startyear)
        + "_"
        + str(endyear)
        + "_rechunked"
        + "_noleap.nc4"
    )
    regression_outfile = variable + dataset + "_regression.nc4"
    detrended_file = variable + dataset + "_detrended.nc4"

min_ts_len = 2  # minimum length of timeseries passed to regression after reduction
sig = 0.95  # significance level to calculate confidence intervals for fits in .
