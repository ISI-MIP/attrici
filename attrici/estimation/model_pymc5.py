"""
ATTRICI model using [PyMC5](https://www.pymc.io/).

The `initialize` function needs to be called once to set PyMC5 settings before using any
other part of the module. See `attrici.detrend`.

This is an updated version of the PyMC3 based version used in the original paper, see
also `attrici.estimation.model_pymc3`.
"""

import atexit
import logging
import shutil
import tempfile
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pymc as pm
from loguru import logger
from pymc.pytensorf import pt

from attrici import distributions
from attrici.estimation.model import Model
from attrici.util import calc_oscillations, collect_windows

# Suppress verbose PyMC logging output
logging.getLogger("pymc").setLevel(logging.WARNING)


def initialize(compile_timeout, use_tmp_compiledir):
    """Initialize the PyMC5 backend."""

    # set PyTensor `compile__timeout`, note that PyTensor's flag uses two underscores
    pt.pytensor.config.compile__timeout = compile_timeout

    # if there are several processes running in parallel, we need to use a temporary
    # directory for each process individually to avoid conflicts
    if use_tmp_compiledir:
        tmpdir = tempfile.mkdtemp()
        pt.pytensor.config.compiledir = tmpdir
        # register a handler that removes the temporary directory on exit:
        atexit.register(shutil.rmtree, tmpdir)

    logger.info("Using PyMC5 version {}", pm.__version__)
    logger.info(
        "PyTensor compilation timeout (in sec): `pytensor.config.compile__timeout`={}",
        pt.pytensor.config.compile__timeout,
    )
    if use_tmp_compiledir:
        logger.info(
            "Using temporary directory for PyTensor compilation: "
            "`pytensor.config.compiledir`={}",
            pt.pytensor.config.compiledir,
        )


def setup_parameter_model(name, parameter, modes=None, window_size=None):
    """
    Setup a parameter model based on the type of parameter.

    Parameters
    ----------
    name : str
        The name of the parameter model.
    parameter : AttriciGLM.Parameter
        The parameter to be used in the model.
    modes : int, optional
        The number of modes to use for the oscillations.
    window_size : int, optional
        The size of the window to use for rolling window fitting.

    Returns
    -------
    AttriciGLMPymc5.PredictorDependentParam or
    AttriciGLMPymc5.PredictorIndependentParam or
    AttriciGLMPymc5.PredictorDependentParamRollingWindow or
    AttriciGLMPymc5.PredictorIndependentParamRollingWindow
        The corresponding model object for the given parameter.
    """
    if modes is not None:
        if parameter.dependent:
            return AttriciGLMPymc5.PredictorDependentParam(name, parameter.link, modes)
        return AttriciGLMPymc5.PredictorIndependentParam(name, parameter.link, modes)

    if window_size is not None:
        if parameter.dependent:
            return AttriciGLMPymc5.PredictorDependentParamRollingWindow(
                name, parameter.link, window_size
            )
        return AttriciGLMPymc5.PredictorIndependentParamRollingWindow(
            name, parameter.link, window_size
        )

    raise ValueError("Exactly one of `modes` and `window_size` must be set")


class AttriciGLMPymc5:
    """
    A class for building and estimating parameters in a Generalized Linear Model with
    PyMC5.
    """

    PRIOR_INTERCEPT_MU = 0
    PRIOR_INTERCEPT_SIGMA = 1
    PRIOR_TREND_MU = 0
    PRIOR_TREND_SIGMA = 0.1

    @dataclass
    class PredictorDependentParam:
        """
        Class for predictor-dependent parameters in the model when using the modes
        based approach.
        Implements equations (1) and (2) of Mengel et al. (2021).

        Attributes
        ----------
        name : str
            The name of the parameter.
        link : Callable
            The link function to be applied.
        modes : int
            The number of modes to use for the oscillations.
        """

        name: str
        link: Callable
        modes: int

        def build_linear_model(self, oscillations, predictor):
            """
            Build a linear model for predictor-dependent parameters.

            Parameters
            ----------
            oscillations : pytensor.Tensor
                The oscillation data for the model.
            predictor : pytensor.Tensor
                The predictor data.

            Returns
            -------
            pytensor.Tensor
                The linear model for the parameter.
            """
            weights_longterm_intercept = pm.Normal(
                f"weights_{self.name}_longterm_intercept",
                mu=AttriciGLMPymc5.PRIOR_INTERCEPT_MU,
                sigma=AttriciGLMPymc5.PRIOR_INTERCEPT_SIGMA,
            )
            weights_fc_intercept = pm.math.concatenate(
                [
                    pm.Normal(
                        f"weights_{self.name}_fc_intercept_{i}",
                        mu=AttriciGLMPymc5.PRIOR_INTERCEPT_MU,
                        sigma=1 / (2 * i + 1),
                        shape=2,
                    )
                    for i in range(self.modes)
                ]
            )

            covariates = pm.math.concatenate(
                [
                    oscillations,
                    pt.tile(predictor[:, None], (1, 2 * self.modes)) * oscillations,
                ],
                axis=1,
            )
            weights_longterm_trend = pm.Normal(
                f"weights_{self.name}_longterm_trend",
                mu=AttriciGLMPymc5.PRIOR_TREND_MU,
                sigma=AttriciGLMPymc5.PRIOR_TREND_SIGMA,
            )
            weights_fc_trend = pm.Normal(
                f"weights_{self.name}_fc_trend",
                mu=AttriciGLMPymc5.PRIOR_TREND_MU,
                sigma=AttriciGLMPymc5.PRIOR_TREND_SIGMA,
                shape=2 * self.modes,
            )
            weights_fc = pm.math.concatenate([weights_fc_intercept, weights_fc_trend])
            return (
                pt.dot(covariates, weights_fc)
                + weights_longterm_intercept
                + weights_longterm_trend * predictor
            )

        def estimate(self, trace, predictor):
            """
            Estimate the parameter based on a trace.

            Parameters
            ----------
            trace : dict
                The trace to estimate from.
            predictor : xarray.DataArray
                The predictor data.

            Returns
            -------
            numpy.ndarray
                The estimated parameter value.
            """
            oscillations = calc_oscillations(predictor.time, self.modes)
            weights_longterm_intercept = trace[
                f"weights_{self.name}_longterm_intercept"
            ]
            weights_fc_intercept = np.concatenate(
                [
                    trace[f"weights_{self.name}_fc_intercept_{i}"]
                    for i in range(self.modes)
                ],
            )
            covariates = np.concatenate(
                [
                    oscillations,
                    np.tile(predictor.values[:, None], (1, 2 * self.modes))
                    * oscillations,
                ],
                axis=1,
            )
            weights_longterm_trend = trace[f"weights_{self.name}_longterm_trend"]
            weights_fc_trend = trace[f"weights_{self.name}_fc_trend"]
            weights_fc = np.concatenate([weights_fc_intercept, weights_fc_trend])
            return self.link(
                np.dot(covariates, weights_fc)
                + weights_longterm_intercept
                + weights_longterm_trend * predictor.values
            )

        def build(self, predictor):
            """
            Build the full model for a predictor-dependent parameter.

            Parameters
            ----------
            predictor : xarray.DataArray
                The predictor data.

            Returns
            -------
            pymc.Deterministic
                The deterministic value for the parameter.
            """
            oscillations = pm.Data(
                f"{self.name}_oscillations",
                calc_oscillations(predictor.time, self.modes),
            )
            predictor = pm.Data(
                f"{self.name}_predictor",
                predictor,
            )
            return pm.Deterministic(
                self.name,
                self.link(self.build_linear_model(oscillations, predictor)),
            )

    @dataclass
    class PredictorIndependentParam:
        """
        Class for predictor-independent parameters in the model when using the modes
        based approach.
        Implements equation (3) of Mengel et al. (2021).

        Attributes
        ----------
        name : str
            The name of the parameter.
        link : Callable
            The link function to be applied.
        modes : int
            The number of modes to use for the oscillations.
        """

        name: str
        link: Callable
        modes: int

        def build_linear_model(self, oscillations):
            """
            Setup a linear model for predictor-independent parameters.

            Parameters
            ----------
            oscillations : pytensor.Tensor
                The oscillation data for the model.

            Returns
            -------
            pytensor.Tensor
                The linear model for the parameter.
            """
            weights_longterm_intercept = pm.Normal(
                f"weights_{self.name}_longterm_intercept",
                mu=AttriciGLMPymc5.PRIOR_INTERCEPT_MU,
                sigma=AttriciGLMPymc5.PRIOR_INTERCEPT_SIGMA,
            )
            weights_fc_intercept = pm.math.concatenate(
                [
                    pm.Normal(
                        f"weights_{self.name}_fc_intercept_{i}",
                        mu=0,
                        sigma=1 / (2 * i + 1),
                        shape=2,
                    )
                    for i in range(self.modes)
                ]
            )
            return (
                pt.dot(oscillations, weights_fc_intercept) + weights_longterm_intercept
            )

        def estimate(self, trace, predictor):
            """
            Estimate the parameter based on a trace.

            Parameters
            ----------
            trace : dict
                The trace to estimate from.
            predictor : xarray.DataArray
                The predictor data.

            Returns
            -------
            numpy.ndarray
                The estimated parameter value.
            """
            oscillations = calc_oscillations(predictor.time, self.modes)
            weights_longterm_intercept = trace[
                f"weights_{self.name}_longterm_intercept"
            ]
            weights_fc_intercept = np.concatenate(
                [
                    trace[f"weights_{self.name}_fc_intercept_{i}"]
                    for i in range(self.modes)
                ],
            )
            return self.link(
                np.dot(oscillations, weights_fc_intercept) + weights_longterm_intercept
            )

        def build(self, predictor):
            """
            Build the full model for a predictor-independent parameter.

            Parameters
            ----------
            predictor : xarray.DataArray
                The predictor data.

            Returns
            -------
            pymc.Deterministic
                The deterministic value for the parameter.
            """
            oscillations = pm.Data(
                f"{self.name}_oscillations",
                calc_oscillations(predictor.time, self.modes),
            )
            return pm.Deterministic(
                self.name,
                self.link(self.build_linear_model(oscillations)),
            )

    @dataclass
    class PredictorDependentParamRollingWindow:
        """
        Class for predictor-dependent parameters in the model when using the rolling
        window approach.

        Attributes
        ----------
        name : str
            The name of the parameter.
        link : Callable
            The link function to be applied.
        window_size : int
            The size of the rolling window.
        """

        name: str
        link: Callable
        window_size: int

        def build_linear_model(self, predictor, daysofyear):
            """
            Build a linear model for predictor-dependent parameters.

            Parameters
            ----------
            predictor : pytensor.Tensor
                The predictor data.

            predictor_shape : tuple
                The shape of the predictor data, should be `(366, window_size)`.

            Returns
            -------
            pytensor.Tensor
                The linear model for the parameter.
            """
            max_dayofyear = 366
            weights_longterm_intercept = pm.Normal(
                f"weights_{self.name}_longterm_intercept",
                mu=[AttriciGLMPymc5.PRIOR_INTERCEPT_MU] * max_dayofyear,
                sigma=[AttriciGLMPymc5.PRIOR_INTERCEPT_SIGMA] * max_dayofyear,
            )

            weights_longterm_trend = pm.Normal(
                f"weights_{self.name}_longterm_trend",
                mu=[AttriciGLMPymc5.PRIOR_INTERCEPT_MU] * max_dayofyear,
                sigma=[AttriciGLMPymc5.PRIOR_INTERCEPT_SIGMA] * max_dayofyear,
            )
            return (
                weights_longterm_intercept[daysofyear - 1]
                + weights_longterm_trend[daysofyear - 1] * predictor
            )

        def estimate(self, trace, predictor):
            """
            Estimate the parameter based on a trace.

            Parameters
            ----------
            trace : dict
                The trace to estimate from.
            predictor : xarray.DataArray
                The predictor data.

            Returns
            -------
            numpy.ndarray
                    The estimated parameter value.
            """
            daysofyear = predictor.time.dt.dayofyear.values
            return self.link(
                trace[f"weights_{self.name}_longterm_intercept"][daysofyear - 1]
                + trace[f"weights_{self.name}_longterm_trend"][daysofyear - 1]
                * predictor.values
            )

        def build(self, predictor):
            """
            Build the full model for a predictor-dependent parameter.

            Parameters
            ----------
            predictor : xarray.DataArray
                The predictor data.

            Returns
            -------
            pymc.Deterministic
                The deterministic value for the parameter.
            """
            predictor_var = pm.Data(
                f"{self.name}_predictor",
                predictor,
            )
            daysofyear = predictor.dayofyear.values
            return pm.Deterministic(
                self.name,
                self.link(self.build_linear_model(predictor_var, daysofyear)),
            )

    @dataclass
    class PredictorIndependentParamRollingWindow:
        """
        Class for predictor-independent parameters in the model when using the rolling
        window approach.

        Attributes
        ----------
        name : str
            The name of the parameter.
        link : Callable
            The link function to be applied.
        window_size : int
            The size of the rolling window.
        """

        name: str
        link: Callable
        window_size: int

        def build_linear_model(self, daysofyear):
            """
            Setup a linear model for predictor-independent parameters.

            Parameters
            ----------
            predictor_shape : tuple
                The shape of the predictor data, should be `(366, window_size)`.

            Returns
            -------
            pytensor.Tensor
                The linear model for the parameter.
            """
            max_dayofyear = 366
            weights_longterm_intercept = pm.Normal(
                f"weights_{self.name}_longterm_intercept",
                mu=[AttriciGLMPymc5.PRIOR_INTERCEPT_MU] * max_dayofyear,
                sigma=[AttriciGLMPymc5.PRIOR_INTERCEPT_SIGMA] * max_dayofyear,
            )
            return weights_longterm_intercept[daysofyear - 1]

        def estimate(self, trace, predictor):
            """
            Estimate the parameter based on a trace.

            Parameters
            ----------
            trace : dict
                The trace to estimate from.
            predictor : xarray.DataArray
                The predictor data.

            Returns
            -------
            numpy.ndarray
                The estimated parameter value.
            """
            daysofyear = predictor.time.dt.dayofyear.values
            return self.link(
                trace[f"weights_{self.name}_longterm_intercept"][daysofyear - 1]
            )

        def build(self, predictor):
            """
            Build the full model for a predictor-independent parameter.

            Parameters
            ----------
            predictor : xarray.DataArray
                The predictor data.

            Returns
            -------
            pymc.Deterministic
                The deterministic value for the parameter.
            """
            return pm.Deterministic(
                self.name,
                self.link(self.build_linear_model(predictor.dayofyear.values)),
            )


class ModelPymc5(Model):
    """Class for building and fitting a model using PyMC5."""

    def __init__(
        self,
        distribution,
        parameters,
        observed,
        predictor,
        modes=None,
        window_size=None,
    ):
        """
        Initialize a PyMC5 model.

        Parameters
        ----------
        distribution : class
            The distribution class to be used (e.g., distributions.Normal).
        parameters : dict
            A dictionary of parameters to be used in the model.
        observed : xarray.DataArray
            The observed data.
        predictor : xarray.DataArray
            The predictor data.
        modes : int, optional
            The number of modes to use for the oscillations.
        window_size : int, optional
            The size of the window to use for rolling window fitting.
        """
        logger.info(f"Using PyMC5 version {pm.__version__}")
        self._distribution_class = distribution
        self._model = pm.Model()
        with self._model:
            self._parameter_models = {
                name: setup_parameter_model(
                    name, parameter, modes=modes, window_size=window_size
                )
                for name, parameter in parameters.items()
            }

            self._window_size = window_size
            if window_size is not None:
                predictor = collect_windows(predictor, window_size)
                observed = collect_windows(observed, window_size)

            if distribution == distributions.BernoulliGamma:
                observed_gamma = observed.sel(time=observed.notnull())

                p = self._parameter_models["p"].build(predictor)
                mu = self._parameter_models["mu"].build(
                    predictor.sel(time=observed_gamma.time)
                )
                nu = self._parameter_models["nu"].build(
                    predictor.sel(time=observed_gamma.time)
                )

                pm.Deterministic("logp", self._model.varlogp)
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
                pm.Deterministic("logp", self._model.varlogp)
                pm.Gamma(
                    "observation",
                    mu=mu,
                    sigma=mu / nu,
                    observed=observed,
                )

            elif distribution == distributions.Normal:
                mu = self._parameter_models["mu"].build(predictor)
                sigma = self._parameter_models["sigma"].build(predictor)
                pm.Deterministic("logp", self._model.varlogp)
                pm.Normal(
                    "observation",
                    mu=mu,
                    sigma=sigma,
                    observed=observed,
                )

            elif distribution == distributions.Beta:
                mu = self._parameter_models["mu"].build(predictor)
                phi = self._parameter_models["phi"].build(predictor)
                pm.Deterministic("logp", self._model.varlogp)
                pm.Beta(
                    "observation",
                    alpha=pm.Deterministic("alpha", mu * phi),
                    beta=pm.Deterministic("beta", (1 - mu) * phi),
                    observed=observed,
                )

            elif distribution == distributions.Weibull:
                alpha = self._parameter_models["alpha"].build(predictor)
                beta = self._parameter_models["beta"].build(predictor)
                pm.Deterministic("logp", self._model.varlogp)
                pm.Weibull("observation", alpha=alpha, beta=beta, observed=observed)

            else:
                raise ValueError(
                    f"Distribution {distribution} not supported"
                )  # pragma: no cover

    def fit(self, progressbar=False, **kwargs):
        """
        Fit the model using PyMC's MAP estimator.

        Parameters
        ----------
        progressbar : bool, optional
            Whether to display a progress bar during fitting.
        **kwargs :
            Additional arguments - not used.

        Returns
        -------
        dict
            A dictionary containing the parameter estimates, the "traces".
        """
        traces = pm.find_MAP(model=self._model, progressbar=progressbar)
        return {
            k: v for k, v in traces.items() if k == "logp" or k.startswith("weights_")
        }

    def estimate_logp(self, trace, progressbar=False, **kwargs):
        """
        Estimate the log-probability based on a trace.

        Parameters
        ----------
        trace : dict
            The trace to estimate from.
        progressbar : bool, optional
            Whether to display a progress bar during sampling.
        **kwargs :
            Additional arguments - not used.

        Returns
        -------
        float
            The estimated log-probability value.
        """
        return trace["logp"]

    def estimate_distribution(self, trace, predictor, progressbar=False, **kwargs):
        """
        Estimate the distribution parameters based on a trace.

        Parameters
        ----------
        trace : dict
            The trace to estimate from.
        predictor : xarray.DataArray
            The predictor data.
        progressbar : bool, optional
            Whether to display a progress bar during sampling.
        **kwargs :
            Additional arguments - not used.

        Returns
        -------
        attrici.distributions.Distribution
            The estimated distribution.
        """
        return self._distribution_class(
            **{
                name: self._parameter_models[name].estimate(trace, predictor)
                for name in self._parameter_models.keys()
            }
        )
