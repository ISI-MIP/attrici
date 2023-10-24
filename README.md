
# ATTRICI - counterfactual climate for impact attribution

Code implementing the methods described in the paper `ATTRICI 1.1 - counterfactual climate for impact attribution` in Geoscientific Model Development. The code is archived at [ZENODO](https://doi.org/10.5281/zenodo.3828914).


## Project Structure
* All general settings are defined in [settings.py](settings.py).
* The code can be run with [run_estimation.py](run_estimation.py) or [run_single_cell.py](run_single_cell.py).
* The probability model for different climate variables is specified in [models.py](attrici/models.py)
* The choice of the probability model for a variable is specified in [estimator.py](attrici/estimator.py)


## Install

Please do

`conda config --add channels conda-forge`

`conda create -c conda-forge -n attrici "pymc>=5" python=3.10.11`

`conda activate attrici`

`conda install netCDF4 pytables matplotlib arviz`

`pip install func_timeout`

You may optionally
`cp config/theanorc ~/.theanorc`


## Usage

The parallelization part to run large datasets is currently taylored to the supercomputer at the Potsdam Institute for Climate Impact Research using the [slurm scheduler](https://slurm.schedmd.com/documentation.html). Generalizing it into a package is ongoing work. We use the GNU compiler as the many parallel compile jobs through jobarrays and JIT compilation conflict with the few Intel licenses.

`module purge`

`module load compiler/gnu/7.3.0`

`conda activate yourenv`

In the root package directory.

`pip install -e .`


Override the conda setting with: `export CXX=g++`

Load input data from [https://data.isimip.org](https://data.isimip.org)

The input for one variable is a single netcdf file containing all time steps. 

Create smoothed gmt time series as predictor using `preprocessing/calc_gmt_by_ssa.py` with adjusted file-paths.

Get auxiliary *tasskew* and *tasrange* time series using `preprocessing/create_tasrange_tasskew.py` with adjusted file-paths.

Adjust `settings.py`

For estimating parameter distributions (above step 1) and smaller datasets

`python run_estimation.py`

For larger datasets, produce a `submit.sh` file via

`python create_submit.py`

Then submit to the slurm scheduler

`sbatch submit.sh`

For merging the single timeseries files to netcdf datasets

`python merge_cfact.py`



### Usage of bash scripts for a more automated processing

Adapt paths or user settings in `settings.py`, `create_runscripts.sh`, `slurm_combined.sh`, `slurm.sh`, `submit_write_netcdf.sh`, `merge_files.sh`, `sanity_checks.sh`
And also adapt file paths in `sanity_check/sanity_check.py`, `sanity_check/merge_files.py` `sanity_check/visual_check.py`


**Example for processing of one variable eg. tas0 for tile 00001**
Create runfolders: `bash create_runscripts.sh 00001`
Run: `sbatch slurm_combined.sh 00001 tas0`
This bash script implements a processing workflow for one variable. The workflow is as follows:
- `slurm_combined.sh` starts `slurm.sh` in the respective runfolder, which will create trace and timeseries files. Number of arrays is limited to 100 running in parallel. To avoid that all job arrays are canceled at once due that cluster resources are needed for another job
- After `slurm.sh` finished, `sanity_checks.sh` starts which checks if all timeseries files are complete and if failing cells occured
- If all sanity checks are successfullly passed, `merge_files.sh` starts which creates a backup containing all trace files
- If all sanity checks are successfullly passed, `submit_write_netcdf.sh` starts which creates the counterfactuals
- If counterfactuals were successfully created, `visual_checks.sh` starts. It provides a glimpse of the final counterfactual netcdf file (filename ending with "*_valid.nc4")
- If `visual_checks.sh` was successfull, `move_final_cfacts.sh` starts inside `output` folder, which copys all counterfactuals, the backup file and the failing_cells.log to the project folder. It removes the source files, so only one version of counterfactuals exists located in the project folder

After checking the log file from `submit_write_netcdf` and having a check of the created plots from `visual_checks.sh` of the final cfact, the timeseries and trace files are removed manually
merge_files.py

## Preprocessing

Example for GSWP3-W5E5 dataset, which is first priority in ISIMIP3a.

`cd preprocess` 
Download decadal data and merge it into one file per variable.
Adjust output paths and
`python merge_data.py`
Approximately 1 hour.

Produce GMT from gridded surface air temperature and use SSA to smooth it.
Use a separate conda env to cover SSA package dependencies.
Adjust output paths and
`python calc_gmt_by_ssa.py`
Approximately less than an hour.

Create tasrange and tasskew from tas variables.
Adjust output paths and
`python create_tasmin_tasmax.py`
Approximately an hour.

For testing smaller dataset, use
`python subset_data.py`
Add sub01 to filenames, if no subsetting is used.

Land-sea file creation
We use the ISIMIP2b land-sea mask to select only land cells for processing.
Smaller datasets through subsetting were created using CDO.

## Postprocessing

For tasmin and tasmax, we do not estimate counterfactual time series individually to avoid large relative errors in the daily temperature range as pointed out by (Piani et al. 2010). Following (Piani et al. 2010), we estimate counterfactuals of the daily temperature range tasrange = tasmax - tasmin and the skewness of the daily temperature tasskew = (tas - tasmin) / tasrange. Use [create_tasmin_tasmax.py](postprocessing/create_tasmin_tasmax.py)
with adjusted paths for the _tas_, _tasrange_ and _tasskew_ counterfactuals.

A counterfactual huss is derived from the counterfacual tas, ps and hurs using the equations of Buck (1981) as described in Weedon et al. (2010). Use [derive_huss.sh](postprocessing/derive_huss.sh)
with adjusted file names and the required time range.


## Credits

We rely on the [pymc3](https://github.com/pymc-devs/pymc3) package for probabilistic programming (Salvatier et al. 2016).

An early version of the code on Bayesian estimation of parameters in timeseries with periodicity in PyMC3 was inspired by [Ritchie Vink's](https://www.ritchievink.com) [post](https://www.ritchievink.com/blog/2018/10/09/build-facebooks-prophet-in-pymc3-bayesian-time-series-analyis-with-generalized-additive-models/) on Bayesian timeseries analysis with additive models.

## License

This code is licensed under GPLv3, see the LICENSE.txt. See commit history for authors.
