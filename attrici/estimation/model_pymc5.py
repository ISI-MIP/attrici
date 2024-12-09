from dataclasses import dataclass

import numpy as np
import pymc as pm
from pymc.pytensorf import pt

from attrici import distributions
from attrici.estimation.model import AttriciGLM, Model


def setup_parameter_model(name, parameter):
    if isinstance(parameter, AttriciGLM.PredictorDependentParam):
        return AttriciGLMPymc5.PredictorDependentParam(name, parameter)
    if isinstance(parameter, AttriciGLM.PredictorIndependentParam):
        return AttriciGLMPymc5.PredictorIndependentParam(name, parameter)
    raise ValueError(f"Parameter type {type(parameter)} not supported")


def calc_oscillations(t, modes):
    t_scaled = (t - t.min()) / (np.timedelta64(365, "D") + np.timedelta64(6, "h"))
    x = (2 * np.pi * (np.arange(modes) + 1)) * t_scaled.to_numpy()[:, None]
    return np.concatenate((np.cos(x), np.sin(x)), axis=1)


class AttriciGLMPymc5:
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


class ModelPymc5(Model):
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

            elif distribution == distributions.Bernoulli:
                p = self._parameter_models["p"].build(predictor)
                pm.Deterministic("logp", self._model.varlogp)
                pm.Bernoulli("observation", p=p, observed=observed)

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
                raise ValueError(f"Distribution {distribution} not supported")

    def fit(self, progressbar=False):
        self.trace = pm.find_MAP(model=self._model, progressbar=progressbar)
        return self.trace

    def estimate_logp(self, progressbar=False):
        with self._model:
            sample = pm.sample_posterior_predictive(
                [self.trace],
                var_names=["logp"],
                progressbar=progressbar,
            )["posterior_predictive"]

            return sample["logp"].values.mean(axis=(0, 1))

    def estimate_distribution(self, predictor, progressbar=False):
        with self._model:
            for parameter_model in self._parameter_models.values():
                parameter_model.set_predictor_data(predictor)

            sample = pm.sample_posterior_predictive(
                [self.trace],
                var_names=list(self._parameter_models.keys()),
                progressbar=progressbar,
            )["posterior_predictive"]

        return self._distribution_class(
            **{
                name: sample[name].values.mean(axis=(0, 1))
                for name in self._parameter_models.keys()
            }
        )
