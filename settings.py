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
variable = "tas"
dataset = "era5"
startyear = 1979
endyear = 2018

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
    to_detrend_file = variable + "_" + dataset + "_test.nc4"
    regression_outfile = variable + "_" + dataset + "_regression_test.nc4"
    detrended_file = variable + "_" + dataset + "_detrended_test.nc4"
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
    regression_outfile = (
        variable
        + "_"
        + dataset
        + "_"
        + str(startyear)
        + "_"
        + str(endyear)
        + "_regression_all.nc4"
    )
    detrended_file = (
        variable
        + "_"
        + dataset
        + "_"
        + str(startyear)
        + "_"
        + str(endyear)
        + "_detrended.nc4"
    )

min_ts_len = 2  # minimum length of timeseries passed to regression after reduction
sig = 0.95  # significance level to calculate confidence intervals for fits in .
