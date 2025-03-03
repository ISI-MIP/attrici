"""
ATTRICI model based on the legacy version of PyMC,
[PyMC3](https://pypi.org/project/pymc3/).

The `initialize` function needs to be called once to import PyMC3 and configure settings
before using any other part of the module. See `attrici.detrend`.

> [!WARNING]
> Note that this module uses the no longer supported or maintained version 3 of PyMC. It
> includes a number of workarounds and patches to allow for compatibility with newer
> versions of Numpy or Python.
> It requires Python 3.11 and is thus not installed by default.

This is based on the version used in the [Attrici
paper](https://doi.org/10.5194/gmd-14-5269-2021).

For a version using PyMC5, see `attrici.estimation.model_pymc5`.
"""

import atexit
import collections
import collections.abc
import inspect
import logging
import os
import shutil
import sys
import tempfile
import warnings
from dataclasses import dataclass

import numpy as np
from loguru import logger

from attrici import distributions
from attrici.estimation.model import AttriciGLM, Model
from attrici.util import calc_oscillations

# monkey patch for newer numpy versions
if not hasattr(np, "asscalar"):
    np.asscalar = np.ndarray.item

np.bool = bool

if not hasattr(collections, "Iterable"):
    collections.Iterable = collections.abc.Iterable

if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec

if sys.version_info >= (3, 12):
    raise ImportError("PyMC3 module is not compatible with Python >3.11")

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    import numpy.distutils as numpy_distutils  # noqa: E402


def initialize(compile_timeout, use_tmp_compiledir):
    """Initialize the PyMC3 backend."""

    def add_theano_flag(flag):
        """
        Add a flag to the Theano configuration (`THEANO_FLAGS` environment variable).

        Parameters
        ----------
        flag : str
            The flag to be added.
        """
        os.environ["THEANO_FLAGS"] = flag + (
            f",{os.environ['THEANO_FLAGS']}" if os.environ.get("THEANO_FLAGS") else ""
        )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        if not hasattr(numpy_distutils.__config__, "blas_opt_info"):
            add_theano_flag("blas.ldflags=")

        # set Theano `compile.timeout`
        add_theano_flag(f"compile.timeout={compile_timeout}")

        # if there are several processes running in parallel, we need to use a temporary
        # directory for each process individually to avoid conflicts
        if use_tmp_compiledir:
            tmpdir = tempfile.mkdtemp()
            add_theano_flag(f"compiledir='{tmpdir}'")
            # register a handler that removes the temporary directory on exit:
            atexit.register(shutil.rmtree, tmpdir)

        # make sure that the packages are available in the global scope
        global pm  # noqa: PLW0603
        global tt  # noqa: PLW0603

        # need to be imported last after numpy and collections have been patched and
        # after theano flags have been set accordingly
        import pymc3 as pm  # noqa: E402
        import theano  # noqa: E402
        import theano.tensor as tt  # noqa: E402

        logger.info("Using PyMC3 version {}", pm.__version__)
        logger.info(
            "Theano compilation timeout (in sec): `theano.config.compile.timeout`={}",
            theano.config.compile.timeout,
        )
        if use_tmp_compiledir:
            logger.info(
                "Using temporary directory for Theano compilation: "
                "`theano.config.compiledir`={}",
                theano.config.compiledir,
            )

        # needed to silence verbose pymc3
        logging.getLogger("pymc3").propagate = False


def setup_parameter_model(name, parameter):
    """
    Setup a parameter model based on the type of parameter.

    Parameters
    ----------
    name : str
        The name of the parameter.
    parameter : Parameter
        The parameter object, either PredictorDependentParam or
        PredictorIndependentParam.

    Returns
    -------
    AttriciGLMPymc3.PredictorDependentParam or AttriciGLMPymc3.PredictorIndependentParam
        A corresponding model based on the type of parameter.

    Raises
    ------
    ValueError
        If the parameter type is not supported.
    """
    if isinstance(parameter, AttriciGLM.PredictorDependentParam):
        return AttriciGLMPymc3.PredictorDependentParam(name, parameter)
    if isinstance(parameter, AttriciGLM.PredictorIndependentParam):
        return AttriciGLMPymc3.PredictorIndependentParam(name, parameter)
    raise ValueError(f"Parameter type {type(parameter)} not supported")


class AttriciGLMPymc3:
    """
    A class for building and estimating parameters in a Generalized Linear Model
    with PyMC3.

    Attributes
    ----------
    PRIOR_INTERCEPT_MU : float
        The prior mean for intercept parameters.
    PRIOR_INTERCEPT_SIGMA : float
        The prior standard deviation for intercept parameters.
    PRIOR_TREND_MU : float
        The prior mean for trend parameters.
    PRIOR_TREND_SIGMA : float
        The prior standard deviation for trend parameters.
    """

    PRIOR_INTERCEPT_MU = 0
    PRIOR_INTERCEPT_SIGMA = 1
    PRIOR_TREND_MU = 0
    PRIOR_TREND_SIGMA = 0.1

    @dataclass
    class PredictorDependentParam:
        """
        A dataclass representing parameters dependent on the predictor.

        Attributes
        ----------
        name : str
            The name of the parameter.
        parameter : AttriciGLM.PredictorDependentParam
            The associated parameter object from AttriciGLM.

        """

        name: str
        parameter: AttriciGLM.PredictorDependentParam

        def build_linear_model(self, oscillations, predictor):
            """
            Setup the linear model for the predictor-dependent parameter.

            Parameters
            ----------
            oscillations : theano.tensor.TensorVariable
                The oscillations (cosine and sine terms) based on the time.
            predictor : theano.tensor.TensorVariable
                The predictor data.

            Returns
            -------
            theano.tensor.TensorVariable
                The linear model as a sum of weighted oscillations and trends.
            """
            weights_longterm_intercept = pm.Normal(
                f"weights_{self.name}_longterm_intercept",
                mu=AttriciGLMPymc3.PRIOR_INTERCEPT_MU,
                sigma=AttriciGLMPymc3.PRIOR_INTERCEPT_SIGMA,
            )
            weights_fc_intercept = pm.math.concatenate(
                [
                    pm.Normal(
                        f"weights_{self.name}_fc_intercept_{i}",
                        mu=AttriciGLMPymc3.PRIOR_INTERCEPT_MU,
                        sigma=1 / (2 * i + 1),
                        shape=2,
                    )
                    for i in range(self.parameter.modes)
                ]
            )

            covariates = pm.math.concatenate(
                [
                    oscillations,
                    tt.tile(predictor[:, None], (1, 2 * self.parameter.modes))
                    * oscillations,
                ],
                axis=1,
            )
            weights_longterm_trend = pm.Normal(
                f"weights_{self.name}_longterm_trend",
                mu=AttriciGLMPymc3.PRIOR_TREND_MU,
                sigma=AttriciGLMPymc3.PRIOR_TREND_SIGMA,
            )
            weights_fc_trend = pm.Normal(
                f"weights_{self.name}_fc_trend",
                mu=AttriciGLMPymc3.PRIOR_TREND_MU,
                sigma=AttriciGLMPymc3.PRIOR_TREND_SIGMA,
                shape=2 * self.parameter.modes,
            )
            weights_fc = pm.math.concatenate([weights_fc_intercept, weights_fc_trend])
            return (
                tt.dot(covariates, weights_fc)
                + weights_longterm_intercept
                + weights_longterm_trend * predictor
            )

        def build(self, predictor):
            """
            Build the deterministic model for the predictor-dependent parameter.

            Parameters
            ----------
            predictor : theano.tensor.TensorVariable
                The predictor data.

            Returns
            -------
            theano.tensor.TensorVariable
                A deterministic output based on the model.
            """
            oscillations = pm.Data(
                f"{self.name}_oscillations",
                calc_oscillations(predictor.time, self.parameter.modes),
            )
            predictor = pm.Data(
                f"{self.name}_predictor",
                predictor,
            )
            return pm.Deterministic(
                self.name,
                self.parameter.link(self.build_linear_model(oscillations, predictor)),
            )

        def set_predictor_data(self, data):
            """
            Set the predictor data for the model.

            Parameters
            ----------
            data : xarray.DataArray
                A xarray containing the predictor data with time.
            """
            pm.set_data(
                {
                    f"{self.name}_oscillations": calc_oscillations(
                        data.time, self.parameter.modes
                    ),
                    f"{self.name}_predictor": data,
                }
            )

    @dataclass
    class PredictorIndependentParam:
        """
        A dataclass representing parameters independent of the predictor.

        Attributes
        ----------
        name : str
            The name of the parameter.
        parameter : AttriciGLM.PredictorIndependentParam
            The associated parameter object from AttriciGLM.
        """

        name: str
        parameter: AttriciGLM.PredictorIndependentParam

        def build_linear_model(self, oscillations):
            """
            Setup the linear model for the predictor-independent parameter.

            Parameters
            ----------
            oscillations : theano.tensor.TensorVariable
                The oscillations (cosine and sine terms).

            Returns
            -------
            theano.tensor.TensorVariable
                The linear model based on oscillations and intercept.
            """
            weights_longterm_intercept = pm.Normal(
                f"weights_{self.name}_longterm_intercept",
                mu=AttriciGLMPymc3.PRIOR_INTERCEPT_MU,
                sigma=AttriciGLMPymc3.PRIOR_INTERCEPT_SIGMA,
            )
            weights_fc_intercept = pm.math.concatenate(
                [
                    pm.Normal(
                        f"weights_{self.name}_fc_intercept_{i}",
                        mu=0,
                        sigma=1 / (2 * i + 1),
                        shape=2,
                    )
                    for i in range(self.parameter.modes)
                ]
            )
            return (
                tt.dot(oscillations, weights_fc_intercept) + weights_longterm_intercept
            )

        def build(self, predictor):
            """
            Build the deterministic model for the predictor-independent parameter.

            Parameters
            ----------
            predictor : theano.tensor.TensorVariable
                The predictor data.

            Returns
            -------
            theano.tensor.TensorVariable
                A deterministic output based on the model.
            """
            oscillations = pm.Data(
                f"{self.name}_oscillations",
                calc_oscillations(predictor.time, self.parameter.modes),
            )
            return pm.Deterministic(
                self.name, self.parameter.link(self.build_linear_model(oscillations))
            )

        def set_predictor_data(self, data):
            """
            Set the predictor data for the model.

            Parameters
            ----------
            data : pd.DataFrame
                A DataFrame containing the predictor data with time.
            """
            pm.set_data(
                {
                    f"{self.name}_oscillations": calc_oscillations(
                        data.time, self.parameter.modes
                    )
                }
            )


class ModelPymc3(Model):
    """A class for building a PyMC3 model for a given distribution and parameter set."""

    def __init__(self, distribution, parameters, observed, predictor):
        """
        Initialize the PyMC3 model.

        Parameters
        ----------
        distribution : class
            The distribution class (e.g., distributions.Bernoulli).
        parameters : dict
            A dictionary of parameter names and their respective parameter objects.
        observed : xarray.DataArray
            The observed data to be used for the model.
        predictor : xarray.DataArray
            The predictor data to be used in the model.
        """
        logger.info(f"Using PyMC3 version {pm.__version__}")
        self._distribution_class = distribution
        self._model = pm.Model()
        with self._model:
            self._parameter_models = {
                name: setup_parameter_model(name, parameter)
                for name, parameter in parameters.items()
            }

            if distribution == distributions.BernoulliGamma:
                observed_gamma = observed.sel(time=observed.notnull())

                p = self._parameter_models["p"].build(predictor)
                mu = self._parameter_models["mu"].build(
                    predictor.sel(time=observed_gamma.time)
                )
                nu = self._parameter_models["nu"].build(
                    predictor.sel(time=observed_gamma.time)
                )

                pm.Deterministic("logp", self._model.logpt)
                pm.Bernoulli(
                    "observation_bernoulli",
                    p=p,
                    observed=np.isnan(observed.values).astype(int),
                )
                pm.Gamma(
                    "observation_gamma",
                    mu=mu,
                    sigma=mu / nu,
                    observed=observed_gamma,
                )

            elif distribution == distributions.Gamma:
                mu = self._parameter_models["mu"].build(predictor)
                nu = self._parameter_models["nu"].build(predictor)
                pm.Deterministic("logp", self._model.logpt)
                pm.Gamma(
                    "observation",
                    mu=mu,
                    sigma=mu / nu,
                    observed=observed,
                )

            elif distribution == distributions.Normal:
                mu = self._parameter_models["mu"].build(predictor)
                sigma = self._parameter_models["sigma"].build(predictor)
                pm.Deterministic("logp", self._model.logpt)
                pm.Normal(
                    "observation",
                    mu=mu,
                    sigma=sigma,
                    observed=observed,
                )

            elif distribution == distributions.Beta:
                mu = self._parameter_models["mu"].build(predictor)
                phi = self._parameter_models["phi"].build(predictor)
                pm.Deterministic("logp", self._model.logpt)
                pm.Beta(
                    "observation",
                    alpha=pm.Deterministic("alpha", mu * phi),
                    beta=pm.Deterministic("beta", (1 - mu) * phi),
                    observed=observed,
                )

            elif distribution == distributions.Weibull:
                alpha = self._parameter_models["alpha"].build(predictor)
                beta = self._parameter_models["beta"].build(predictor)
                pm.Deterministic("logp", self._model.logpt)
                pm.Weibull("observation", alpha=alpha, beta=beta, observed=observed)

            else:
                raise ValueError(f"Distribution {distribution} not supported")

    def fit(self, progressbar=False, **kwargs):
        """
        Fit the model using maximum a posteriori (MAP) estimation.

        Parameters
        ----------
        progressbar : bool, optional
            Whether to display a progress bar during fitting, by default False.

        Returns
        -------
        pymc3.model.Model
            The fitted PyMC3 model.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            traces = pm.find_MAP(model=self._model, progressbar=progressbar)
            return {
                k: v
                for k, v in traces.items()
                if k == "logp" or k.startswith("weights_")
            }

    def estimate_logp(self, trace, progressbar=False, **kwargs):
        """
        Estimate the log-probability of the model.

        Parameters
        ----------
        trace : pymc3.backends.base.MultiTrace
            The trace of the posterior samples.
        progressbar : bool, optional
            Whether to display a progress bar during the estimation, by default False.

        Returns
        -------
        float
            The estimated log-probability.
        """
        with self._model:
            sample = pm.sample_posterior_predictive(
                [trace],
                var_names=["logp"],
                samples=1,
                progressbar=progressbar,
            )

            return sample["logp"].mean(axis=0)

    def estimate_distribution(self, trace, predictor, progressbar=False, **kwargs):
        """
        Estimate the distribution for the given predictor.

        Parameters
        ----------
        trace : pymc3.backends.base.MultiTrace
            The trace of the posterior samples.
        predictor : xarray.DataArray
            The predictor data.
        progressbar : bool, optional
            Whether to display a progress bar during the estimation, by default False.

        Returns
        -------
        attrici.distributions.Distribution
            The estimated distribution based on the model.
        """
        with self._model:
            for parameter_model in self._parameter_models.values():
                parameter_model.set_predictor_data(predictor)

            sample = pm.sample_posterior_predictive(
                [trace],
                var_names=list(self._parameter_models.keys()),
                samples=1,
                progressbar=progressbar,
            )

        return self._distribution_class(
            **{
                name: sample[name].mean(axis=0)
                for name in self._parameter_models.keys()
            }
        )
