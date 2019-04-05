# Detrending

Scripts for detrending reanalysis data

Preprocessing:

Concatenate input files: Run preprocessing/merge_data.sh
gets variable from settings.py
Change other settings in file!
might request user input!

Remove leap days (29.2.): Run preprocessing/remove_leap_days.sh
runs all variables
Change settings in file!

Rechunk dataset for faster access to individual timeseries: Run preprocessing/rechunk.sh
runs all variables
Change settings in file!

Create small test data set: Run preprocessing/create_test_data.py
gets variable from settings.py

Calculate global mean temperature (destination variable): Run preprocessing/calc_gmt_by_ssa.py
gets variable from settings.py


Main program:

Adjust settings.py to your needs!
Run submit.sh via slurm


Post processing:

Split data into files of 10 years: Run postprocessing/split_data.sh


Utility:

Some tests for input data


Visualization:

Some helpful functions to visualize data are in idetrend/visualization.py
