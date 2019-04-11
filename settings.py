import getpass
import scipy.special as sp

user = getpass.getuser()

# this will hopefully avoid hand editing paths everytime.
# fill further for convenience.
if user == "mengel":
    data_dir = "/home/mengel/data/20190306_IsimipDetrend/"
else:
    data_dir = "/home/bschmidt/temp/gswp3/"

#  handle job specifications
n_jobs = 16  # number of childprocesses created by the job

# if test=True use smaller test dataset
test = True
variable = "tas"
dataset = "gswp3"
startyear = 1901
endyear = 2010

days_of_year = 365

# this dictionary sets the transformations we need to
# do for certain variables. these come in tuples
# [transform, inverse_transform]
transforms = {"pr":[sp.logit, sp.expit]}

gmt_file = "test_ssa_gmt.nc4"
base_file = (
    variable
    + "_rechunked_"
    + dataset
    + "_"
    + str(startyear)
    + "_"
    + str(endyear)
    + "_noleap.nc4"
)

if test:
    to_detrend_file = "test_data_" + variable + ".nc4"
    regression_outfile = variable + "_regression_test.nc4"
else:
    to_detrend_file = (
        variable
        + "_rechunked_"
        + dataset
        + "_"
        + str(startyear)
        + "_"
        + str(endyear)
        + "_noleap.nc4"
    )
    regression_outfile = variable + "_regression_all.nc4"
