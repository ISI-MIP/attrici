import collections
import collections.abc
import inspect
import logging
import os
import sys
import warnings
from dataclasses import dataclass

import numpy as np
from loguru import logger

from attrici import distributions
from attrici.estimation.model import AttriciGLM, Model

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

    if not hasattr(numpy_distutils.__config__, "blas_opt_info"):
        os.environ["THEANO_FLAGS"] = "blas.ldflags=" + (
            f",{os.environ['THEANO_FLAGS']}" if os.environ.get("THEANO_FLAGS") else ""
        )

    # need to be imported last after numpy and collections have been patched
    import pymc3 as pm  # noqa: E402
    import theano.tensor as tt  # noqa: E402

logging.getLogger("pymc3").propagate = False  # needed to silence verbose pymc3


def initialize():
    logger.info("Using PyMC3 version {}", pm.__version__)


def setup_parameter_model(name, parameter):
    if isinstance(parameter, AttriciGLM.PredictorDependentParam):
        return AttriciGLMPymc3.PredictorDependentParam(name, parameter)
    if isinstance(parameter, AttriciGLM.PredictorIndependentParam):
        return AttriciGLMPymc3.PredictorIndependentParam(name, parameter)
    raise ValueError(f"Parameter type {type(parameter)} not supported")


def calc_oscillations(t, modes):
    t_scaled = (t - t.min()) / (np.timedelta64(365, "D") + np.timedelta64(6, "h"))
    x = (2 * np.pi * (np.arange(modes) + 1)) * t_scaled.to_numpy()[:, None]
    return np.concatenate((np.cos(x), np.sin(x)), axis=1)


class AttriciGLMPymc3:
    PRIOR_INTERCEPT_MU = 0
    PRIOR_INTERCEPT_SIGMA = 1
    PRIOR_TREND_MU = 0
    PRIOR_TREND_SIGMA = 0.1

    @dataclass
    class PredictorDependentParam:
        name: str
        parameter: AttriciGLM.PredictorDependentParam

        def build_linear_model(self, oscillations, predictor):
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
        name: str
        parameter: AttriciGLM.PredictorIndependentParam

        def build_linear_model(self, oscillations):
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
            oscillations = pm.Data(
                f"{self.name}_oscillations",
                calc_oscillations(predictor.time, self.parameter.modes),
            )
            return pm.Deterministic(
                self.name, self.parameter.link(self.build_linear_model(oscillations))
            )

        def set_predictor_data(self, data):
            pm.set_data(
                {
                    f"{self.name}_oscillations": calc_oscillations(
                        data.time, self.parameter.modes
                    )
                }
            )


class ModelPymc3(Model):
    def __init__(
        self,
        distribution,
        parameters,
        observed,
        predictor,
    ):
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

            elif distribution == distributions.Bernoulli:
                p = self._parameter_models["p"].build(predictor)
                pm.Deterministic("logp", self._model.logpt)
                pm.Bernoulli("observation", p=p, observed=observed)

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

    def fit(
        self,
        progressbar=False,
        **kwargs,
    ):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            traces = pm.find_MAP(model=self._model, progressbar=progressbar)
            return {
                k: v
                for k, v in traces.items()
                if k == "logp" or k.startswith("weights_")
            }

    def estimate_logp(
        self,
        trace,
        progressbar=False,
        **kwargs,
    ):
        with self._model:
            sample = pm.sample_posterior_predictive(
                [trace],
                var_names=["logp"],
                samples=1,
                progressbar=progressbar,
            )

            return sample["logp"].mean(axis=0)

    def estimate_distribution(
        self,
        trace,
        predictor,
        progressbar=False,
        **kwargs,
    ):
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
