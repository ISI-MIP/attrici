import getpass

#  import numpy as np
#  import idetrend.utility as u

user = getpass.getuser()

# this will hopefully avoid hand editing paths everytime.
# fill further for convenience.
if user == "mengel":
    data_dir = "/home/mengel/data/20190306_IsimipDetrend/"
elif user == "bschmidt":
    data_dir = "/home/bschmidt/temp/gswp3/"

#  handle job specifications
n_jobs = 16  # number of childprocesses created by the job
regtype = "bayes"  # regression type: 'bayes' or 'linear'

# if test=True use smaller test dataset
test = True  # use to run on test dataset
variable = "tas"  # select variable to detrend
dataset = "era5"  # select dataset to run on
startyear = 1979  # select startyear
endyear = 2018  # select endyear
# have to be in filenames
calendar = "gregorian"  # 'gregorian' or 'noleap' implemented

################### For Bayesian ############

# model run settings
debug = True  # use to turn on debug settings
mpi = True  # use True to run on cluster (multiple nodes)
init = "jitter+adapt_diag"  # init method for nuts sampler
ntunes = 800  # number of draws to tune model
ndraws = 1000  # number of sampling draws per chaiin
nchains = 5  # number of chains to calculate (min 2 to check for convergence)
ncores_per_job = 2  # number of cores to use for one gridpoint
# automatically set to 1 for mpi (last line)
progressbar = False  # print progress in output (.err file for mpi)
live_plot = False  # show live plot (does not work yet)

# set model parameters
modes = 1  # number of modes for fourier series of model
linear_mu = 1  # mean of prior for linear model
linear_sigma = 10  # sd of prior for linear model
sigma_beta = 0.5  # beta parameter of halfcauchy sd of model
smu = 1  # seasonal prior mean
stmu = 0.5  # trend in season prior mean
sps = 20  # seasonality prior scale (sd)
stps = 20  # trend in season scale (sd)


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
    to_detrend_file = (
        variable
        + "_"
        + dataset
        + "_"
        + str(startyear)
        + "_"
        + str(endyear)
        + "_"
        + calendar
        + "_test.nc4"
    )
    regression_outfile = (
        variable
        + "_"
        + dataset
        + "_"
        + str(startyear)
        + "_"
        + str(endyear)
        + "_"
        + regtype
        + "_test.nc4"
    )
    detrended_file = (
        variable
        + "_"
        + dataset
        + "_"
        + str(startyear)
        + "_"
        + str(endyear)
        + "_"
        + regtype
        + "_detrended_test.nc4"
    )
else:
    to_detrend_file = (
        variable
        + "_"
        + dataset
        + "_"
        + str(startyear)
        + "_"
        + str(endyear)
        + "_"
        + calendar
        + "_rechunked"
        + ".nc4"
    )
    regression_outfile = (
        variable
        + "_"
        + dataset
        + "_"
        + str(startyear)
        + "_"
        + str(endyear)
        + "_"
        + regtype
        + ".nc4"
    )
    detrended_file = (
        variable
        + "_"
        + dataset
        + "_"
        + str(startyear)
        + "_"
        + str(endyear)
        + "_"
        + regtype
        + "_detrended.nc4"
    )

min_ts_len = 2  # minimum length of timeseries passed to regression after reduction
sig = 0.95  # significance level to calculate confidence intervals for fits in .

if calendar == "gregorian":
    days_of_year = 365.25
elif calendar == "noleap":
    days_of_year = 365

if mpi:
    ncores_per_job = 1
if debug:
    regression_outfile = "debug_reg_out.nc4"
