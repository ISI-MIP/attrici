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

import numpy as np
import pymc as pm
from loguru import logger
from pymc.pytensorf import pt

from attrici import distributions
from attrici.estimation.model import AttriciGLM, Model
from attrici.util import calc_oscillations

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


def setup_parameter_model(name, parameter):
    """
    Setup a parameter model based on the type of parameter.

    Parameters
    ----------
    name : str
        The name of the parameter model.
    parameter : AttriciGLM.PredictorDependentParam or
                AttriciGLM.PredictorIndependentParam
        The parameter to be used in the model.

    Returns
    -------
    AttriciGLMPymc5.PredictorDependentParam or AttriciGLMPymc5.PredictorIndependentParam
        The corresponding model class for the given parameter.

    Raises
    ------
    ValueError
        If the parameter type is not supported.
    """
    if isinstance(parameter, AttriciGLM.PredictorDependentParam):
        return AttriciGLMPymc5.PredictorDependentParam(name, parameter)
    if isinstance(parameter, AttriciGLM.PredictorIndependentParam):
        return AttriciGLMPymc5.PredictorIndependentParam(name, parameter)
    raise ValueError(
        f"Parameter type {type(parameter)} not supported"
    )  # pragma: no cover


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
        Class for predictor-dependent parameters in the model.

        Attributes
        ----------
        name : str
            The name of the parameter.
        parameter : AttriciGLM.PredictorDependentParam
            The corresponding parameter from AttriciGLM.
        """

        name: str
        parameter: AttriciGLM.PredictorDependentParam

        def build_linear_model(self, oscillations, predictor):
            """
            Build a linear model for predictor-dependent parameters.

            Parameters
            ----------
            oscillations : pytensor.Tensor
                The oscillation data for the model.
            predictor : xarray.DataArray
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
                    for i in range(self.parameter.modes)
                ]
            )

            covariates = pm.math.concatenate(
                [
                    oscillations,
                    pt.tile(predictor[:, None], (1, 2 * self.parameter.modes))
                    * oscillations,
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
                shape=2 * self.parameter.modes,
            )
            weights_fc = pm.math.concatenate([weights_fc_intercept, weights_fc_trend])
            return (
                pt.dot(covariates, weights_fc)
                + weights_longterm_intercept
                + weights_longterm_trend * predictor
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
            Set the data for the predictor.

            Parameters
            ----------
            data : xarray.DataArray
                The data to be used for the predictor.
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
        Class for predictor-independent parameters in the model.

        Attributes
        ----------
        name : str
            The name of the parameter.
        parameter : AttriciGLM.PredictorIndependentParam
            The corresponding parameter from AttriciGLM.
        """

        name: str
        parameter: AttriciGLM.PredictorIndependentParam

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
                    for i in range(self.parameter.modes)
                ]
            )
            return (
                pt.dot(oscillations, weights_fc_intercept) + weights_longterm_intercept
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
                calc_oscillations(predictor.time, self.parameter.modes),
            )
            return pm.Deterministic(
                self.name, self.parameter.link(self.build_linear_model(oscillations))
            )

        def set_predictor_data(self, data):
            """
            Sets the data for the predictor.

            Parameters
            ----------
            data : xarray.DataArray
                The data to be used for the predictor.
            """
            pm.set_data(
                {
                    f"{self.name}_oscillations": calc_oscillations(
                        data.time, self.parameter.modes
                    )
                }
            )


class ModelPymc5(Model):
    """Class for building and fitting a model using PyMC5."""

    def __init__(self, distribution, parameters, observed, predictor):
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

        """
        logger.info(f"Using PyMC5 version {pm.__version__}")
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
            Additional arguments passed to
            [`pm.find_MAP`](https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.find_MAP.html).

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
            Additional arguments passed to
            [`pm.sample_posterior_predictive`](https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.sample_posterior_predictive.html).

        Returns
        -------
        float
            The estimated log-probability value.
        """
        with self._model:
            sample = pm.sample_posterior_predictive(
                [trace],
                var_names=["logp"],
                progressbar=progressbar,
            )["posterior_predictive"]

            return sample["logp"].values.mean(axis=(0, 1))

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
            Additional arguments passed to `pm.sample_posterior_predictive`.

        Returns
        -------
        attrici.distributions.Distribution
            The estimated distribution.
        """
        with self._model:
            for parameter_model in self._parameter_models.values():
                parameter_model.set_predictor_data(predictor)

            sample = pm.sample_posterior_predictive(
                [trace],
                var_names=list(self._parameter_models.keys()),
                progressbar=progressbar,
            )["posterior_predictive"]

        return self._distribution_class(
            **{
                name: sample[name].values.mean(axis=(0, 1))
                for name in self._parameter_models.keys()
            }
        )
