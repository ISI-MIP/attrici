# ATTRICI - counterfactual climate for impact attribution

Code implementing the methods described in the paper [`ATTRICI 1.1 - counterfactual climate for impact attribution`](https://doi.org/10.5194/gmd-14-5269-2021) in Geoscientific Model Development. The code is archived at [ZENODO](https://doi.org/10.5281/zenodo.3828914).


## Project Structure
* The probability model for different climate variables is specified in [variables.py](attrici/variables.py)
* The choice of the probability model for a variable is specified in [estimator.py](attrici/estimator.py)


## Install instructions for the development version

Clone the repository and checkout the current development branch.

Create a virtual environment.

```
python3 -m venv env
```

Activate the environment (adjust if using a non-default shell)

```
source env/bin/activate
```

Install ATTRICI as a local development version with dev dependencies included

```
pip install -e .[dev]
```


## Usage

See [USAGE.md](USAGE.md) for examples.

## Credits

We rely on the [PyMC](https://www.pymc.io/) package for probabilistic programming (Salvatier et al. 2016).

An early version of the code on Bayesian estimation of parameters in timeseries with periodicity in PyMC3 was inspired by [Ritchie Vink's](https://www.ritchievink.com) [post](https://www.ritchievink.com/blog/2018/10/09/build-facebooks-prophet-in-pymc3-bayesian-time-series-analyis-with-generalized-additive-models/) on Bayesian timeseries analysis with additive models.

## License

This code is licensed under GPLv3, see the [LICENSE.txt](LICENSE.txt). See commit history for authors.
