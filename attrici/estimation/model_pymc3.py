import collections
import collections.abc
import inspect
import logging

import numpy as np

from attrici import distributions
from attrici.estimation.model import (
    AttriciGLM,
    Model,
)

# monkey patch for newer numpy versions
if not hasattr(np, "asscalar"):
    np.asscalar = np.ndarray.item

np.bool = bool

if not hasattr(collections, "Iterable"):
    collections.Iterable = collections.abc.Iterable

if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec

import pymc3 as pm  # noqa
import theano.tensor as tt  # noqa

logging.getLogger("pymc3").propagate = False  # needed to silence verbose pymc3


def setup_parameter_model(name, parameter):
    if isinstance(parameter, AttriciGLM.PredictorDependentParam):
        return AttriciGLMPymc3(name, parameter, is_independent=False)
    if isinstance(parameter, AttriciGLM.PredictorIndependentParam):
        return AttriciGLMPymc3(name, parameter, is_independent=True)
    raise ValueError(f"Parameter type {type(parameter)} not supported")


class AttriciGLMPymc3:
    def __init__(self, name, parameter, is_independent):
        self.name = name
        self.parameter = parameter
        self.is_independent = is_independent

    def _calc_oscillations(self, t):
        t_scaled = (t - t.min()) / (np.timedelta64(365, "D") + np.timedelta64(6, "h"))
        x = (2 * np.pi * (np.arange(self.parameter.modes) + 1)) * t_scaled.to_numpy()[
            :, None
        ]
        return np.concatenate((np.cos(x), np.sin(x)), axis=1)

    def _calc_linear_model(self, predictor):
        oscillations = pm.Data(
            f"{self.name}_oscillations", self._calc_oscillations(predictor.time)
        )

        weights_longterm_intercept = pm.Normal(
            f"weights_{self.name}_longterm_intercept", mu=0, sd=1
        )
        weights_fc_intercept = pm.math.concatenate(
            [
                pm.Normal(
                    f"weights_{self.name}_fc_intercept_{i}",
                    mu=0,
                    sd=1 / (2 * i + 1),
                    shape=2,
                )
                for i in range(self.parameter.modes)
            ]
        )
        if self.is_independent:
            return (
                tt.dot(oscillations, weights_fc_intercept) + weights_longterm_intercept
            )

        predictor = pm.Data(f"{self.name}_predictor", predictor)

        covariates = pm.math.concatenate(
            [
                oscillations,
                tt.tile(predictor[:, None], (1, 2 * self.parameter.modes))
                * oscillations,
            ],
            axis=1,
        )
        weights_longterm_trend = pm.Normal(
            f"weights_{self.name}_longterm_trend", mu=0, sd=0.1
        )
        weights_fc_trend = pm.Normal(
            f"weights_{self.name}_fc_trend",
            mu=0,
            sd=0.1,
            shape=2 * self.parameter.modes,
        )
        weights_fc = pm.math.concatenate([weights_fc_intercept, weights_fc_trend])
        return (
            tt.dot(covariates, weights_fc)
            + weights_longterm_intercept
            + weights_longterm_trend * predictor
        )

    def build(self, predictor):
        return pm.Deterministic(
            self.name, self.parameter.link(self._calc_linear_model(predictor))
        )

    def set_predictor_data(self, data):
        pm.set_data({f"{self.name}_oscillations": self._calc_oscillations(data.time)})
        if not self.is_independent:
            pm.set_data({f"{self.name}_predictor": data})


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
                    sigma=pm.Deterministic("sigma", mu / nu),
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
                    sigma=pm.Deterministic("sigma", mu / nu),
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

    def fit(self, progressbar=False):
        self.trace = pm.find_MAP(model=self._model, progressbar=progressbar)
        return self.trace

    def estimate_logp(self, progressbar=False):
        with self._model:
            sample = pm.sample_posterior_predictive(
                [self.trace],
                samples=1,
                var_names=["logp"],
                progressbar=progressbar,
            )
            return sample["logp"].mean(axis=0)

    def estimate_distribution(self, predictor, progressbar=False):
        with self._model:
            for parameter_model in self._parameter_models.values():
                parameter_model.set_predictor_data(predictor)
            sample = pm.sample_posterior_predictive(
                [self.trace],
                samples=1,
                var_names=list(self._parameter_models.keys()),
                progressbar=progressbar,
            )

        return self._distribution_class(
            **{
                name: sample[name].mean(axis=0)
                for name in self._parameter_models.keys()
            }
        )
