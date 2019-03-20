import getpass

user = getpass.getuser()

# this will hopefully avoid hand editing paths everytime.
# fill further for convenience.
if user == "mengel":
	data_dir = "/home/mengel/data/20190306_IsimipDetrend/"
else:
	data_dir = "/home/bschmidt/temp/gswp3/"

#  handle job specifications
n_jobs = 16  # number of childprocesses created by the job

variable = 'tas'


gmt_file = 'test_ssa_gmt.nc4'
to_detrend_file = 'test_data_' + variable + '.nc4'
base_file = variable + '_rm_rechunked_gswp3_1901_2010.nc4'
regression_outfile = variable + '_regression.nc4'
