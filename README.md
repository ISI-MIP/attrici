# ISI-CFACT

ISI-CFACT produces counterfactual climate data from past datasets for the ISIMIP project.

## Idea
Counterfactual climate is a hypothetical climate in a world without climate change.
For impact models, such climate should stay as close as possible to the observed past,
as we aim to compare impact events of the past (for which we have data) to the events in the counterfactual. The difference between past impacts and counterfactual impacts is a proxy for the impacts caused by climate change. We run the following steps:

1. We approximate the change in past climate through a model with three parts. Long-term trend, an ever-repeating yearly cycle, and a trend in the yearly cycle. Trends are induced by global mean temperature change. We use a Bayesian approach to estimate all parameters of the model and their dependencies at once, here implemented through pymc3. Yearly cycle and trend in yearly cycles are approximated through a finite number of modes, which are periodic in the year. The parameter distributions tell us which part of changes in the variables can be explained through global mean temperature as a direct driver.

2. We do quantile mapping to map each value from the observed dataset to a value that we expect it would have been without the climate-induced trend. Our hierachical model approach provides us with a time evolution of our distribution through the time evolution of a gmt-dependent parameter.
We first this time-evolving distribution to map each value to its quantile in this time evolving distribution.
We then use the distribution from a reference period in the beginning of our dataset where we assume that climate change did not play a role, to remap the quantile to value of the variable. This value is our counterfactual value. Quantile mapping is different for each day of the year because our model is sensitive to the yearly cycle and the trend in the yearly cycle

The following graph illustrates the approach. Grey is the original data, red is our estimation of change. Blue is the original data minus the parts that were estimated to driven by global mean temperature change.

![Counterfactual example](image01.png)

## Example

See [here](examples/tas_example.ipynb) for a notebook leading you through the basic steps.

## Usage

This code is currently taylored to run on the supercomputer at the Potsdam Institute for Climate Impact Research. Generalizing it into a package is ongoing work. We use the GNU compiler as the many parallel compile jobs through jobarrays and JIT compilation conflict with the few Intel licenses.

`module purge`

`module load compiler/gnu/7.3.0`

`conda activate yourenv`

Override the conda setting with: `export CXX=g++`

Adjust `settings.py`

For estimating parameter distributions (above step 1) and smaller datasets

`python run_estimation.py`

For larger datasets, produce a `submit.sh` file via

`python create_submit.py`

Then submit to the slurm scheduler

`sbatch submit.sh`

For merging the single timeseries files to netcdf datasets

`python merge_cfact.py`

### Running multiple instances at once


`conda activate yourenv`

In the root package directory.

`pip install -e .`

Copy the `settings.py`, `run_estimation.py`, `merge_cfact.py` and `submit.sh` to a separate directory,
for example `myrunscripts`. Adjust `settings.py` and `submit.sh`, in particular the output directoy, and continue as in Usage.

## Install

We use the jobarray feature of slurm to run many jobs in parallel.
The configuration is very much tailored to the PIK supercomputer at the moment. Please do

`conda config --add channels conda-forge`

`conda create -n isi-cfact pymc3==3.7 python==3.7`

`conda activate isi-cfact`

`conda install netCDF4 pytables matplotlib arviz`

`pip install func_timeout`

You may optionally
`cp config/theanorc ~/.theanorc`

## Comments for each variable

#### tas
data checked
Works using Normal distribution

#### rlds
data checked
Works using Normal distribution
Needs a restart to finish some hanging runs

#### psl / ps
data checked
Works using Normal distribution

#### rsds
Deviationg approach from Lange et al. 2019, using Normal distribution
This is because the yearly cycle is handled inherently here, so no need for specific treatment.
FIXME: produces unrealistic incoming radiation below zero. Needs a different approach

#### hurs (relative humidity)
data checked
With Beta distribution, working
Needs to be rerun so some holes are filled.

GSWP: needs preprocessing to rename from rhs to hurs, and mask invalid values below zero:

```
ncrename -O -v rhs,hurs fname1.nc fname2.nc

cdo setrtomiss,-1e20,0 fname2.nc fname3.nc
```

#### tasskew
data checked
Works using Beta distribution

#### prsnratio
Beta distribution
Snow included in GSWP3

#### tasrange
With Rice distribution
ADVI introduces strong positive trend.
Possible issue: use real mu, not nu for quantile mapping.

#### tasmin
Constructed from tas, tasskew and tasrange
To do in postprocessing

#### tasmax
Constructed from tas, tasskew and tasrange
To do in postprocessing

#### pr
Gamma distribution
Does not remove all regional trends with NUTS.
Fails with ADVI.
Low latitudes are particularly most difficult.

#### wind
Works using Weibull distribution
FIXME: does not seem to detrend. Seems we rather chose the parameter that adjusted the variability range


## Credits

The code on Bayesian estimation of parameters in timeseries with periodicity in PyMC3 is inspired and adopted from [Ritchie Vink's](https://www.ritchievink.com) [post](https://www.ritchievink.com/blog/2018/10/09/build-facebooks-prophet-in-pymc3-bayesian-time-series-analyis-with-generalized-additive-models/) on Bayesian timeseries analysis with additive models.

## License

This code is licensed under GPLv3, see the LICENSE.txt. See commit history for authors.

