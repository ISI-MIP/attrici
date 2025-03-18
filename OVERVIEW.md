# Overview

## Modules

In the `attrici` folder you will find the package's modules:

- For the Command Line Interface (CLI), the `attrici.cli` module, which is called when you run the `attrici` command in the terminal. It includes the particular commands (i.e., the first argument after `attrici` in the terminal) from the `attrici.commands` module. For each of these commands, the respective module implements an `add_parser` function that defines the command's arguments and options via `argparse`. The actual command is implemented in a function that is called when the command is executed, the `run` function.

- The `attrici.preprocessing` module contains functions for preprocessing the data — for example, a function to calculate the smoothed global mean temperature (GMT) using Singular Spectrum Analysis (SSA), which itself is implemented in the `attrici.ssa` module. This, in turn, calls functions from the `attrici.vendored` module, which contains the SSA implementation from the [pyts](https://pyts.readthedocs.io/) package (the code was included with minor modifications to simplify installation and work around memory issues on some systems).

- The `attrici.detrend` module contains the main detrending functionality. The `detrend` function is the main entry point for detrending a timeseries. It takes a timeseries and a variable name as input, and returns the detrended timeseries. The function uses the `attrici.variables` module to get the variable's distribution and the `attrici.estimation` module to estimate the distribution parameters. For details see [Statistical model and detrending](#statistical-model-and-detrending).

- The `attrici.util` module contains utility functions that are used throughout the package, such as functions for timing or for handling provenance metadata.

## Libraries

Throughout the package, we use the following general libraries:

- [loguru](https://github.com/delgan/loguru) for logging: Just `from loguru import logger`and use `logger.info`, `logger.debug`, etc. to log messages. The log messages are written to the console if the respective log level is set in the configuration file or on the command line.
- [tqdm](https://tqdm.github.io) for progress bars: Use `tqdm` as a context manager to wrap an iterable and display a progress bar (e.g. see `attrici.commands.derive_huss`).
- [xarray](https://xarray.dev) for handling multi-dimensional arrays: Use `xarray` to read and write NetCDF files, and to manipulate the data. Most importantly, `xarray` provides a `DataArray` object that is similar to a `pandas` `DataFrame`, but for multi-dimensional data. This is particularly useful for keeping track of some metadata such as the respective geographical cell's latitude and longitude when passing timeseries data between functions.

Additionally, different estimators for the statistical model use different solvers:

- [PyMC3](https://pypi.org/project/pymc3)
- [PyMC5](https://www.pymc.io)
- [Scipy](https://scipy.org/)

## Statistical model and detrending

For ATTRICI's main functionality, the detrending of timeseries data, a statistical model is built to estimate the distribution parameters of the data. The detrending process is as follows (separate for each geographical cell and climate variable):

1. Scale the timeseries data for normalization (e.g. to the unit interval; *specific for the respective climate variable*)
2. Setup the statistical model (*variable-specific*)
3. Set external values for the statistical model — the predictor data (e.g. the smoothed global mean temperature (GMT))
4. Fit the model to the (scaled) timeseries data
5. Estimate the timeseries of distribution parameters for the input timeseries — deriving the "factual" distribution including the trend estimated from the predictor data
6. Estimate the timeseries of distribution parameters for the counterfactual by setting the predictor to `0` — deriving the "counterfactual" distribution without the estimated trend
7. Quantile map the observed value in the reference distribution to the respective quantile value in the counterfactual distribution
8. Re-scale the normalized quantile-mapped values to the original scale (*variable-specific*)

This functionality is brought together in the `attrici.detrend` module, where the `detrend` function is the main entry point for detrending a timeseries. For the specific steps:

- The variable specific parts are implemented in the `attrici.variables` module, where a class is defined for each climate variable, see [Variable classes](#variable-classes).
- The estimation of the statistical model is implemented in the `attrici.estimation` module, where different optimization frameworks are implemented, see [Statistical model estimation](#statistical-model-estimation).
- Both make use of the `attrici.distributions` module, where the probability distributions are defined, see [Distributions](#distributions).

### Variable classes

For each climate variable, a class is defined in `attrici.variables` that handles the specific rescaling and quantile mapping. The class should derive from the `attrici.variables.Variable` class and implement the following methods:

- `__init__(self, data)` to initialise the variable. This function should validate the input data, e.g. bounds and units, by calling the variable's `validate` method. Also, the function should set the variable's `y_scaled` attribute to the scaled data as well as store information necessary for rescaling in the `scaling`attribute (a dictionary).
- `validate(self, data)` to validate the input data, e.g. bounds and units. This method should raise a `ValueError` if the data is not valid.
- `create_model(self, statistical_model, predictor, modes)` to create the statistical model. The `statistical_model` argument is the statistical model class to be used for creating the model (a subclass of `attrici.estimation.model.AttriciGLM`). The `predictor` argument is the predictor data used for the model. The `modes` argument is the number of seasonal (Fourier) modes to be considered in the model. The method should return an instance of the created model. In particular, when instantiating the model, the variable-specific distribution and parameters are to be set. For instance, for near-surface air temperature, the distribution is a normal distribution with mean and standard deviation as parameters:

    ```python
    def create_model(self, statistical_model_class, predictor, modes):
        observation = self.y_scaled.sel(time=self.y_scaled.notnull()).sel(
            time=predictor.time
        )
        return statistical_model_class(
            distribution=distributions.Normal,
            parameters={
                "mu": AttriciGLM.PredictorDependentParam(link=identity, modes=modes),
                "sigma": AttriciGLM.PredictorIndependentParam(link=np.exp, modes=modes),
            },
            observed=observation,
            predictor=predictor.sel(time=observation.time),
        )
    ```

- `quantile_mapping(self, distribution_ref, distribution_cfact)` to map the (scaled) data (in the `y_scaled` attribute) to the respective quantile in the reference distribution. This method should return the data mapped to the respective quantile in the counterfactual distribution. This method is implemented by default in the `Variable` class to simply lookup each time step's quantile according to the reference distribution and return the respective value in the counterfactual distribution:

    ```python
    distribution_cfact.invcdf(distribution_ref.cdf(self.y_scaled))
    ```

    It should be overwritten if the variable-specific quantile mapping is different.

- `rescale(self, scaled_data)` to rescale the data. This method should use the information stored in the `scaling` attribute to rescale the data. The method should return the re-scaled data.

### Statistical model estimation

Each statistical model is implemented as a class in the `attrici.estimation.model` module deriving from the `attrici.estimation.model.Model` class. It should implement the following methods:

- `__init__(self, distribution, parameters, observed, predictor)` to initialise the model. The `distribution` argument is the distribution class to be used for the model (e.g. `distributions.Normal`). The `parameters` argument is a dictionary describing how to handle the parameters of the distribution (see below). The `observed` argument is the observed data. The `predictor` argument is the predictor data. For instance, the `attrici.estimation.model_pymc5.ModelPymc5` class internally initializes the `PyMC` model according to the given parameters.
- `fit(self)` to fit the model to the observed data. This method should return a dictionary with the estimated model parameters — the "traces". The latter is used to load the model from cache if the model has already been fitted.
- `estimate_logp(self, trace)` to estimate the log-probability of the observed data given the estimated model parameters.
- `estimate_distribution(self, trace, predictor)` to estimate the fitted distribution given the estimated model parameters in `trace` and the `predictor` data. This method should return an instance of a `distributions.Distribution` class with the estimated distribution parameters.

The inner model for each parameter of the distribution is described by an instance of the respective parameter class. For now, only the `attrici.estimation.model.AttriciGLM.PredictorDependentParam` and `attrici.estimation.model.AttriciGLM.PredictorIndependentParam` classes are available representing the ATTRICI Generalized Linear Model (GLM) for a parameter that depends on the predictor and a parameter that does not depend on the predictor, respectively. The `link` argument is the link function to be used for the parameter, and the `modes` argument is the number of modes to be considered in the model. For instance, in the `attrici.estimation.model_pymc5.ModelPymc5` class, these classes are remapped to data structures describing the inner model in `PyMC5` structures (e.g. `pm.Deterministic`).

### Distributions

The probability distributions are defined in the `attrici.distributions` module. Each distribution should derive from the `attrici.distributions.Distribution` class. An instance of a distribution represents a distribution for a given timeseries of parameters. Each distribution should be marked as a`@dataclass`, be initialized with the parameters of the distribution, and implement the following methods:

- `cdf(self, y)` to calculate the cumulative distribution function (CDF), i.e. mapping values to the respective quantile.
- `invcdf(self, quantile)` to calculate the inverse cumulative distribution function (inverse CDF), i.e. mapping quantiles to the respective value.
- `expectation(self)` to calculate the expectation of the distribution.

All of these functions should be vectorized, i.e. they should work with array-like structures as input and return array-like structures as output. The actual distribution parameters (which are single-valued floats are timeseries) are stored in the respective attributes of the distribution instance. For instance, the `distributions.Normal` class stores the mean and standard deviation in the `mu` and `sigma` attributes, respectively.

## Further documentation

See also the docs of the respective modules in the API docs and the [example notebooks](https://github.com/ISI-MIP/attrici/tree/main/notebooks) (`notebooks` folder in the repository).