import getpass
from pathlib import Path

user = getpass.getuser()

# this will hopefully avoid hand editing paths everytime.
# fill further for convenience.
if user == "mengel":
    conda_path = "/home/mengel/anaconda3/envs/pymc3/"
    data_dir = "/home/mengel/data/20190306_IsimipDetrend/"
    # data_dir = "/p/tmp/mengel/isimip/isi-cfact"
    log_dir = "./log"

elif user == "bschmidt":
    conda_path = "/home/bschmidt/.conda/envs/mpi_py3"
    data_dir = "/home/bschmidt/temp/isi-cfact/"
    log_dir = "./output"

input_dir = Path(data_dir) / "input"
# make output dir same as cwd. Helps if running more than one job.
output_dir = Path(data_dir) / "output" / Path.cwd().name

# max time in sec for sampler for a single grid cell.
timeout = 60 * 60
# tas, tasrange pr, prsn, prsnratio, ps, rlds, wind, hurs
variable = "tasrange"  # select variable to detrend
# number of modes for fourier series of model
modes = [2,2,2,2]
# out of full, no_gmt_trend, no_gmt_cycle_trend
sigma_model = "full"
prior_type = ["full","long_term"]
subset = 1  # only use every subset datapoint for bayes estimation for speedup
# use the estimated variability in qm
scale_variability = False
# out of "watch+wfdei", "GSWP3", "GSWP3+ERA5"
# use a dataset with only subset spatial grid points for testing
lateral_sub = 80

dataset = "GSWP3"  # select dataset to run on

# start and end date are the time period used to construct
# the reference distribution for quantile mapping.
# take care that period encompasses a leap year
qm_ref_period = ["1901-01-01", "1904-12-31"]
report_mu_sigma = True

gmt_file = dataset.lower() + "_ssa_gmt.nc4"
# source_file = variable + "_" + dataset + "_sub.nc4"
source_file = variable + "_" + dataset + "_sub" + str(lateral_sub) + ".nc4"
params_file = variable + "_" + dataset + "_parameters.nc4"
cfact_file = variable + "_" + dataset + "_cfactual.nc4"
trend_file = variable + "_" + dataset + "_trend.nc4"
# .h5 or .csv
storage_format = ".h5"

# model run settings
tune = 100  # number of draws to tune model
draws = 1000  # number of sampling draws per chain
chains = 2  # number of chains to calculate (min 2 to check for convergence)

# number of cores to use for one gridpoint
# submitted jobs will have ncores_per_job=1 always.
ncores_per_job = 2
progressbar = True  # print progress in output (.err file for mpi)


#### settings for create_submit.py
# number of parallel jobs through jobarray
# needs to be divisor of number of grid cells
njobarray = 64
