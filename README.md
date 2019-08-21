# ISI-CFACT

ISI-CFACT produces counterfactual climate data from past datasets for the ISIMIP project.

## Idea
Counterfactual climate is a hypothetical climate in a world without climate change.
For impact models, such climate should stay as close as possible to the observed past,
as we aim to compare impact events of the past (for which we have data) to the events in the counterfactual. The difference between past impacts and counterfactual impacts is a proxy for the impacts caused by climate change. We run the following steps:

1. We approximate the change in past climate through a model with three parts. Long-term trend, an ever-repeating yearly cycle, and a trend in the yearly cycle. Trends are induced by global mean temperature change. We use a Bayesian approach to estimate all parameters of the model and their dependencies at once, here implemented through pymc3. Yearly cycle and trend in yearly cycles are approximated through a finite number of modes, which are periodic in the year. The parameter distributions tell us which part of changes in the variables can be explained through global mean temperature as a direct driver.

2. We subtract the estimated trends from global mean temperature change from the observed data. This provides a counterfactual, which comes without the trends easily explained by global mean temperature change.


The following graph illustrates the approach. Grey is the original data, red is our estimation of change. Blue is the original data minus the parts that were estimated to driven by global mean temperature change.

![Counterfactual example](image01.png)

## Comments for each variable

#### tas
Works fine using normal distribution

#### tasmin
Constructed from tas, tasskew and tasrange
To do in postprocessing

#### tasmax
Constructed from tas, tasskew and tasrange
To do in postprocessing

#### tasrange
CHECK

#### tasskew
works using Beta distribution

#### rsds
Deviationg approach from Lange et al. 2019, using Normal distribution
This is because the yearly cycle is handled inherently here, so no need for specific treatment.

#### rlds
Works using Normal distribution

#### pr
Works with Gamma distribution

#### wind
Works currently using a Normal distribution, may need to switch to lower-bounded distribution.

#### psl / ps
Works using Normal distribution

#### prsnratio
Works using beta distribution

#### hurs (relative humidity)
With Beta distribution, working

## Usage

This code is currently taylored to run on the supercomputer at the Potsdam Institute for Climate Impact Research. Generalizing it into a package is ongoing work.

`module load intel/2019.3`

`conda activate yourenv`

Adjust `settings.py`

For estimating parameter distributions (above step 1) and smaller datasets

`python run_bayes.py`

For larger datasets, produce a `submit.sh` file via

`python create_submit.py`

Then submit to the slurm scheduler

`sbatch submit.sh`

For producing counterfactuals (above step 2)

`sbatch merge_submit.py`

### Running multiple instances at once


`conda activate yourenv`

In the root package directory.

`pip install -e .`


Copy the `settings.py`, `run_bayes.py` and `submit.sh` to a separate directory,
for example `myrunscripts`. Adjust `settings.py` and `submit.sh`, in particular the output directoy, and continue as in Usage.

## Install

We use the jobarray feature of slurm to run many jobs in parallel. We use the intel-optimized python libraries for performance. The configuration is very much tailored to the PIK supercomputer at the moment. Please do

`conda config --add channels intel`

`conda env create --name yourenv -f config/environment.yml`

You may also optionally

`cp config/theanorc ~/.theanorc`

To enable parallel netCDF output, you need a netCDF4-python module, compiled against a mpi-enabled netcdf-c as well as hdf5 library. To this date, there is no such module available on conda's well known channels, this should be compiled as follows:

1. Download a version from Unidata: https://github.com/Unidata/netcdf4-python/releases <br />
  In this case, 1.5.1.2, and unpack.<br />

2. Create a conda environment (or install into an environment the does not have netcdf4 module installed yet), based on Intel, with mpi4py and numpy<br />

   `module load anaconda/5.0.0_py3` <br />
   `conda create -n yourenv -c intel mpi4py numpy`<br />

3. activate: `source activate yourenv`<br />

4. load an Intel module (for the compiler)<br />
   `module load intel/2018.3`

5. load a recent parallel NetCDF4 module and HDF5 module<br />

   `module load netcdf-c/4.6.2/intel/parallel`<br />
   `module load hdf5/1.10.2/intel/parallel`

6. in the unpacked netcdf4-parallel directory from step 1:<br />

   `CC=mpiicc python setup.py install`<br />

7. confirm module installed:<br />

   `conda list | grep netcdf4` should output:<br />

   netcdf4            1.5.1.2           pypi_0    pypi

To use:<br />
`module load anaconda/5.0.0_py3`<br />
`source activate par_io`<br />

To test:<br />
`export I_MPI_FABRICS=shm:shm` # only to be set for testing on login nodes, not for submitted jobs <br />
`python -c "from netCDF4 import Dataset; Dataset('test.nc', 'w', parallel=True)"`<br />

## Credits

The code on Bayesian estimation of parameters in timeseries with periodicity in PyMC3 is inspired and adopted from [Ritchie Vink's](https://www.ritchievink.com) [post](https://www.ritchievink.com/blog/2018/10/09/build-facebooks-prophet-in-pymc3-bayesian-time-series-analyis-with-generalized-additive-models/) on Bayesian timeseries analysis with additive models.

## License

This code is licensed under GPLv3, see the LICENSE.txt. See commit history for authors.

