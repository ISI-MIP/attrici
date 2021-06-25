
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

`conda create -n attrici pymc3==3.7 python==3.7`

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

### Handle several runs with different settings

Copy the `settings.py`, `run_estimation.py`, `merge_cfact.py` and `submit.sh` to a separate directory,
for example `myrunscripts`. Adjust `settings.py` and `submit.sh`, in particular the output directoy, and continue as in Usage.

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
