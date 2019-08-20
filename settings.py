import getpass
from pathlib import Path

user = getpass.getuser()

# this will hopefully avoid hand editing paths everytime.
# fill further for convenience.
if user == "mengel":
    conda_path = "/home/mengel/anaconda3/envs/pymc3/"
    data_dir = "/home/mengel/data/20190306_IsimipDetrend/"
    # data_dir = "/p/projects/tumble/mengel/isimip/20190612_Isicfact_TestData/"
    # data_dir = "/p/tmp/mengel/isimip/isi-cfact"
    log_dir = "./log"

elif user == "bschmidt":
    conda_path = "/home/bschmidt/.conda/envs/mpi_py3"
    data_dir = "/home/bschmidt/temp/isi-cfact/"
    log_dir = "./output"

input_dir = Path(data_dir) / "input"
# make output dir same as cwd. Helps if running more than one job.
output_dir = Path(data_dir) / "output" / Path.cwd().name

# number of parallel jobs through jobarray
# used through submit.sh, needs to be divisor of number of grid cells
njobarray = 64

# tas, tasrange pr, prsn, prsnratio, ps, rlds, wind
variable = "rsds"  # select variable to detrend
dataset = "watch+wfdei"  # select dataset to run on

gmt_file = dataset + "_ssa_gmt.nc4"
# source_file = variable + "_" + dataset + "_sub.nc4"
source_file = variable + "_" + dataset + "_sub80.nc4"
params_file = variable + "_" + dataset + "_parameters.nc4"
cfact_file = variable + "_" + dataset + "_cfactual.nc4"
trend_file = variable + "_" + dataset + "_trend.nc4"
# .h5 or .csv
storage_format = ".h5"


# length of the gregorian year, as used in GSWP3 and ERA5 data.
days_of_year = 365.25

# model run settings
init = "jitter+adapt_diag"  # init method for nuts sampler
tune = 1000  # number of draws to tune model
draws = 2000  # number of sampling draws per chain
chains = 2  # number of chains to calculate (min 2 to check for convergence)
subset = 5  # only use every subset datapoint for bayes estimation for speedup
lateral_sub = (
    80
)  # subset dataset (lateral) by sampling every [number] grid point in both dimensions

# number of cores to use for one gridpoint
# submitted jobs will have ncores_per_job=1 always.
ncores_per_job = 2
# automatically set to 1 for mpi (last line)
progressbar = True  # print progress in output (.err file for mpi)
live_plot = False  # show live plot (does not work yet)

# parameters for fourier modes and priors
modes = 3  # number of modes for fourier series of model
linear_mu = 0.2  # mean of prior for linear model
linear_sigma = 0.8  # sd of prior for linear model
sigma_beta = 0.5  # beta parameter of halfcauchy sd of model
smu = 0  # seasonal prior mean
stmu = 0  # trend in season prior mean
sps = 0.1  # seasonality prior scale (sd)
stps = 0.5  # trend in season scale (sd)
