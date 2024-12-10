from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from scipy import stats
from scipy.optimize import minimize

from attrici import distributions
from attrici.estimation.model import AttriciGLM, Model


def setup_parameter_model(name, parameter, params_first_index):
    if isinstance(parameter, AttriciGLM.PredictorDependentParam):
        return AttriciGLMScipy.PredictorDependentParam(
            name=name, parameter=parameter, params_first_index=params_first_index
        )
    if isinstance(parameter, AttriciGLM.PredictorIndependentParam):
        return AttriciGLMScipy.PredictorIndependentParam(
            name=name, parameter=parameter, params_first_index=params_first_index
        )
    raise ValueError(f"Parameter type {type(parameter)} not supported")


def calc_oscillations(t, modes):
    t_scaled = (t - t.min()) / (np.timedelta64(365, "D") + np.timedelta64(6, "h"))
    x = (2 * np.pi * (np.arange(modes) + 1)) * t_scaled.values[:, None]
    return np.concatenate((np.cos(x), np.sin(x)), axis=1)


class ParameterScipy:
    pass


class AttriciGLMScipy:
    PRIOR_INTERCEPT_MU = 0
    PRIOR_INTERCEPT_SIGMA = 1
    PRIOR_TREND_MU = 0
    PRIOR_TREND_SIGMA = 0.1

    @dataclass
    class PredictorDependentParam(ParameterScipy):
        name: str
        params_first_index: int
        parameter: AttriciGLM.PredictorDependentParam
        covariates: Any = None

        def get_initial_params(self):
            return np.zeros(2 + 4 * self.parameter.modes)

        def estimate(self, params):
            weights_longterm_intercept = params[self.params_first_index]
            weights_longterm_trend = params[self.params_first_index + 1]
            weights_fc_intercept = params[
                self.params_first_index + 2 : self.params_first_index
                + 2
                + 2 * self.parameter.modes
            ]
            weights_fc_trend = params[
                self.params_first_index
                + 2
                + 2 * self.parameter.modes : self.params_first_index
                + 2
                + 4 * self.parameter.modes
            ]

            logp_prior = stats.norm.logpdf(
                weights_longterm_intercept,
                loc=AttriciGLMScipy.PRIOR_INTERCEPT_MU,
                scale=AttriciGLMScipy.PRIOR_INTERCEPT_SIGMA,
            )
            logp_prior += stats.norm.logpdf(
                weights_longterm_trend,
                loc=AttriciGLMScipy.PRIOR_TREND_MU,
                scale=AttriciGLMScipy.PRIOR_TREND_SIGMA,
            )
            logp_prior += np.sum(
                [
                    stats.norm.logpdf(
                        weights_fc_intercept[i],
                        loc=AttriciGLMScipy.PRIOR_INTERCEPT_MU,
                        scale=1 / (2 * i + 1),
                    )
                    for i in range(self.parameter.modes)
                ]
            )
            logp_prior += np.sum(
                [
                    stats.norm.logpdf(
                        weights_fc_trend[i],
                        loc=AttriciGLMScipy.PRIOR_TREND_MU,
                        scale=AttriciGLMScipy.PRIOR_TREND_SIGMA,
                    )
                    for i in range(self.parameter.modes)
                ]
            )

            weights_fc = np.concatenate([weights_fc_intercept, weights_fc_trend])
            return (
                np.dot(self.covariates, weights_fc)
                + weights_longterm_intercept
                + weights_longterm_trend * self.predictor
            ), logp_prior

        def set_predictor_data(self, data):
            oscillations = calc_oscillations(data.time, self.parameter.modes)
            self.covariates = np.concatenate(
                [
                    oscillations,
                    np.tile(data.values[:, None], (1, 2 * self.parameter.modes))
                    * oscillations,
                ],
                axis=1,
            )
            self.predictor = data

    @dataclass
    class PredictorIndependentParam(ParameterScipy):
        name: str
        params_first_index: int
        parameter: AttriciGLM.PredictorIndependentParam
        oscillations: Any = None

        def get_initial_params(self):
            return np.zeros(1 + 2 * self.parameter.modes)

        def estimate(self, params):
            weights_longterm_intercept = params[self.params_first_index]
            weights_fc_intercept = params[
                self.params_first_index + 1 : self.params_first_index
                + 1
                + 2 * self.parameter.modes
            ]
            logp_prior = stats.norm.logpdf(
                weights_longterm_intercept,
                loc=AttriciGLMScipy.PRIOR_INTERCEPT_MU,
                scale=AttriciGLMScipy.PRIOR_INTERCEPT_SIGMA,
            )
            logp_prior += np.sum(
                [
                    stats.norm.logpdf(
                        weights_fc_intercept[i],
                        loc=AttriciGLMScipy.PRIOR_INTERCEPT_MU,
                        scale=1 / (2 * i + 1),
                    )
                    for i in range(self.parameter.modes)
                ]
            )

            return (
                self.parameter.link(
                    np.dot(self.oscillations, weights_fc_intercept)
                    + weights_longterm_intercept
                ),
                logp_prior,
            )

        def set_predictor_data(self, data):
            self.oscillations = calc_oscillations(data.time, self.parameter.modes)


@dataclass
class DistributionScipy:
    logpdf: Callable
    parameters: dict[str, ParameterScipy]
    observed: Any

    def log_likelihood(self, params):
        res = 0
        params_dict = {}
        for name, parameter in self.parameters.items():
            p, logp = parameter.estimate(params)
            res += logp
            params_dict[name] = p
        return res + np.sum(self.logpdf(self.observed, **params_dict))


def distribution_beta(x, mu, phi):
    return stats.beta.logpdf(x, mu * phi, (1 - mu) * phi)


def distributions_gamma(x, mu, nu):
    return stats.gamma.logpdf(x, nu**2, scale=mu / nu**2)


class ModelScipy(Model):
    def __init__(
        self,
        distribution,
        parameters,
        observed,
        predictor,
    ):
        self._distribution_class = distribution
        self._distributions = []
        self._initial_params = np.asarray([])
        self._parameter_models = {}
        for name, parameter in parameters.items():
            p = setup_parameter_model(name, parameter, len(self._initial_params))
            self._initial_params = np.concatenate(
                [self._initial_params, p.get_initial_params()]
            )
            self._parameter_models[name] = p

        if distribution == distributions.BernoulliGamma:
            observed_gamma = observed.sel(time=observed.notnull())

            p = self._parameter_models["p"]
            p.set_predictor_data(predictor)
            mu = self._parameter_models["mu"]
            mu.set_predictor_data(predictor.sel(time=observed_gamma.time))
            nu = self._parameter_models["nu"]
            nu.set_predictor_data(predictor.sel(time=observed_gamma.time))

            self._distributions.append(
                DistributionScipy(
                    logpdf=distributions_gamma,
                    parameters={"mu": mu, "nu": nu},
                    observed=observed_gamma,
                )
            )
            self._distributions.append(
                DistributionScipy(
                    logpdf=stats.bernoulli.logpmf,
                    parameters={"p": p},
                    observed=np.isnan(observed.values).astype(int),
                )
            )

        elif distribution == distributions.Bernoulli:
            p = self._parameter_models["p"]
            p.set_predictor_data(predictor)
            self._distributions.append(
                DistributionScipy(
                    logpdf=stats.bernoulli.logpmf,
                    parameters={"p": p},
                    observed=observed,
                )
            )

        elif distribution == distributions.Gamma:
            mu = self._parameter_models["mu"]
            mu.set_predictor_data(predictor)
            nu = self._parameter_models["nu"]
            nu.set_predictor_data(predictor)
            self._distributions.append(
                DistributionScipy(
                    logpdf=distributions_gamma,
                    parameters={"mu": mu, "nu": nu},
                    observed=observed,
                )
            )

        elif distribution == distributions.Normal:
            mu = self._parameter_models["mu"]
            mu.set_predictor_data(predictor)
            sigma = self._parameter_models["sigma"]
            sigma.set_predictor_data(predictor)
            self._distributions.append(
                DistributionScipy(
                    logpdf=stats.norm.logpdf,
                    parameters={"loc": mu, "scale": sigma},
                    observed=observed,
                )
            )

        elif distribution == distributions.Beta:
            mu = self._parameter_models["mu"]
            mu.set_predictor_data(predictor)
            phi = self._parameter_models["phi"]
            phi.set_predictor_data(predictor)
            self._distributions.append(
                DistributionScipy(
                    logpdf=distribution_beta,
                    parameters={"mu": mu, "phi": phi},
                    observed=observed,
                )
            )

        elif distribution == distributions.Weibull:
            alpha = self._parameter_models["alpha"]
            alpha.set_predictor_data(predictor)
            beta = self._parameter_models["beta"]
            beta.set_predictor_data(predictor)
            self._distributions.append(
                DistributionScipy(
                    logpdf=stats.weibull_min.logpdf,
                    parameters={"c": alpha, "scale": beta},
                    observed=observed,
                )
            )

        else:
            raise ValueError(f"Distribution {distribution} not supported")

    def fit(self, **_):
        result = minimize(
            lambda params: -sum(d.log_likelihood(params) for d in self._distributions),
            self._initial_params,
            method="L-BFGS-B",
        )
        self.logp = -result.fun
        self.trace = result.x
        return self.trace

    def estimate_logp(self, **_):
        return self.logp

    def estimate_distribution(self, predictor, **_):
        params = {}
        for name, parameter_model in self._parameter_models.items():
            parameter_model.set_predictor_data(predictor)
            params[name], _ = parameter_model.estimate(self.trace)

        return self._distribution_class(**params)
