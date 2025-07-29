"""ATTRICI model using Scipy."""

from dataclasses import dataclass
from typing import Callable

import numpy as np
import xarray as xr

# ArrayLike might not look nice with pdoc https://github.com/mitmproxy/pdoc/issues/420
from numpy.typing import ArrayLike
from scipy import stats
from scipy.optimize import minimize

from attrici import distributions
from attrici.estimation.model import Model
from attrici.util import calc_oscillations


def setup_parameter_model(
    name, parameter, params_first_index, modes=None, window_size=None
):
    """
    Setup a parameter model based on the type of parameter.

    Parameters
    ----------
    name : str
        The name of the parameter model.
    parameter : AttriciGLM.Parameter
        The parameter to be used in the model.
    params_first_index : int
        The index of the first parameter.
    modes : int, optional
        The number of modes to use for the oscillations.
    window_size : int, optional
        The size of the window to use for rolling window fitting.

    Returns
    -------
    AttriciGLMScipy.PredictorDependentParam or AttriciGLMScipy.PredictorIndependentParam
        The corresponding model object for the given parameter.
    """
    if modes is not None:
        if parameter.dependent:
            return AttriciGLMScipy.PredictorDependentParam(
                name=name,
                link=parameter.link,
                modes=modes,
                params_first_index=params_first_index,
            )
        return AttriciGLMScipy.PredictorIndependentParam(
            name=name,
            link=parameter.link,
            modes=modes,
            params_first_index=params_first_index,
        )

    if window_size is not None:
        raise NotImplementedError

    raise ValueError("Exactly one of `modes` and `window_size` must be set")


class ParameterScipy:
    """Base class for parameter models in Scipy."""

    pass


class AttriciGLMScipy:
    """Class for handling parameter estimation using Scipy."""

    PRIOR_INTERCEPT_MU = 0
    PRIOR_INTERCEPT_SIGMA = 1
    PRIOR_TREND_MU = 0
    PRIOR_TREND_SIGMA = 0.1

    @dataclass
    class PredictorDependentParam(ParameterScipy):
        """
        Class for handling predictor dependent parameters.
        Implements equations (1) and (2) of Mengel et al. (2021).

        Attributes
        ----------
        name : str
            Name of the parameter.
        params_first_index : int
            Index of the first parameter.
        link : Callable
            The link function to be applied.
        modes : int
            The number of modes to use for the oscillations.
        covariates : ArrayLike or None
            Covariates for the parameter.
        """

        name: str
        params_first_index: int
        link: Callable
        modes: int
        covariates: ArrayLike | None = None

        def get_initial_params(self):
            """
            Get the initial parameters.

            Returns
            -------
            ndarray
                Initial parameters as a numpy array.
            """
            return np.zeros(2 + 4 * self.modes)

        def estimate(self, params):
            """
            Estimate the parameter values.

            Parameters
            ----------
            params : ArrayLike
                Array of parameter values.

            Returns
            -------
            tuple
                Estimated values and log prior probability.
            """
            weights_longterm_intercept = params[self.params_first_index]
            weights_longterm_trend = params[self.params_first_index + 1]
            weights_fc_intercept = params[
                self.params_first_index + 2 : self.params_first_index
                + 2
                + 2 * self.modes
            ]
            weights_fc_trend = params[
                self.params_first_index + 2 + 2 * self.modes : self.params_first_index
                + 2
                + 4 * self.modes
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
                    for i in range(self.modes)
                ]
            )
            logp_prior += np.sum(
                [
                    stats.norm.logpdf(
                        weights_fc_trend[i],
                        loc=AttriciGLMScipy.PRIOR_TREND_MU,
                        scale=AttriciGLMScipy.PRIOR_TREND_SIGMA,
                    )
                    for i in range(self.modes)
                ]
            )

            weights_fc = np.concatenate([weights_fc_intercept, weights_fc_trend])
            return (
                np.dot(self.covariates, weights_fc)
                + weights_longterm_intercept
                + weights_longterm_trend * self.predictor
            ), logp_prior

        def set_predictor_data(self, data):
            """
            Set the predictor data.

            Parameters
            ----------
            data : xarray.DataArray
                Array of predictor data.
            """
            oscillations = calc_oscillations(data.time, self.modes)
            self.covariates = np.concatenate(
                [
                    oscillations,
                    np.tile(data.values[:, None], (1, 2 * self.modes)) * oscillations,
                ],
                axis=1,
            )
            self.predictor = data

    @dataclass
    class PredictorIndependentParam(ParameterScipy):
        """
        Class for handling predictor independent parameters.
        Implements equation (3) of Mengel et al. (2021).

        Attributes
        ----------
        name : str
            Name of the parameter.
        params_first_index : int
            Index of the first parameter.
        link : Callable
            The link function to be applied.
        modes : int
            The number of modes to use for the oscillations.
        oscillations : ArrayLike or None
            Oscillations for the parameter.
        """

        name: str
        params_first_index: int
        link: Callable
        modes: int
        oscillations: ArrayLike | None = None

        def get_initial_params(self):
            """
            Get the initial parameters.

            Returns
            -------
            ndarray
                Initial parameters as a numpy array.
            """
            return np.zeros(1 + 2 * self.modes)

        def estimate(self, params):
            """
            Estimate the parameter values.

            Parameters
            ----------
            params : ArrayLike
                Array of parameter values.

            Returns
            -------
            tuple
                Estimated values and log prior probability.
            """
            weights_longterm_intercept = params[self.params_first_index]
            weights_fc_intercept = params[
                self.params_first_index + 1 : self.params_first_index
                + 1
                + 2 * self.modes
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
                    for i in range(self.modes)
                ]
            )

            return (
                self.link(
                    np.dot(self.oscillations, weights_fc_intercept)
                    + weights_longterm_intercept
                ),
                logp_prior,
            )

        def set_predictor_data(self, data):
            """
            Set the predictor data.

            Parameters
            ----------
            data : xarray.DataArray
                Array of predictor data.
            """
            self.oscillations = calc_oscillations(data.time, self.modes)


@dataclass
class DistributionScipy:
    """
    Class for handling distributions in Scipy.

    Attributes
    ----------
    logpdf : Callable
        Log probability density function.
    parameters : dict[str, ParameterScipy]
        Dictionary of parameters.
    observed : xarray.DataArray
        Observed data.
    """

    logpdf: Callable
    parameters: dict[str, ParameterScipy]
    observed: xr.DataArray

    def log_likelihood(self, params):
        """
        Calculate the log likelihood.

        Parameters
        ----------
        params : ArrayLike
            Array of parameter values.

        Returns
        -------
        float
            Log likelihood value.
        """
        res = 0
        params_dict = {}
        for name, parameter in self.parameters.items():
            p, logp = parameter.estimate(params)
            res += logp
            params_dict[name] = p
        return res + np.sum(self.logpdf(self.observed, **params_dict))


def distribution_beta(x, mu, phi):
    """
    Calculate the log probability density function of a Beta distribution.

    Parameters
    ----------
    x : ArrayLike
        Observed data.
    mu : float
        Mean of the Beta distribution.
    phi : float
        Dispersion parameter of the Beta distribution.

    Returns
    -------
    ndarray
        Log probability density values.
    """
    return stats.beta.logpdf(x, mu * phi, (1 - mu) * phi)


def distributions_gamma(x, mu, nu):
    """
    Calculate the log probability density function of a Gamma distribution.

    Parameters
    ----------
    x : ArrayLike
        Observed data.
    mu : float
        Mean of the Gamma distribution.
    nu : float
        Shape parameter of the Gamma distribution.

    Returns
    -------
    ndarray
        Log probability density values.
    """
    return stats.gamma.logpdf(x, nu**2, scale=mu / nu**2)


class ModelScipy(Model):
    """Class for handling model estimation using Scipy."""

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
        Initialize a Model using Scipy.

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
        self._distribution_class = distribution
        self._distributions = []
        self._initial_params = np.asarray([])
        self._parameter_models = {}
        for name, parameter in parameters.items():
            p = setup_parameter_model(
                name,
                parameter,
                len(self._initial_params),
                modes=modes,
                window_size=window_size,
            )
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
            raise ValueError(
                f"Distribution {distribution} not supported"
            )  # pragma: no cover

    def fit(self, **kwargs):
        """
        Fit the model using maximum likelihood estimation.

        Parameters
        ----------
        **kwargs :
            Additional arguments - not used.

        Returns
        -------
        dict
            Dictionary containing the log probability and parameter values.
        """
        result = minimize(
            lambda params: -sum(d.log_likelihood(params) for d in self._distributions),
            self._initial_params,
            method="L-BFGS-B",
        )
        return {
            "logp": -result.fun,
            "params": result.x,
        }

    def estimate_logp(self, trace, **kwargs):
        """
        Estimate the log probability.

        Parameters
        ----------
        trace : dict
            Dictionary containing the trace of parameter values.
        **kwargs :
            Additional arguments - not used.

        Returns
        -------
        float
            Log probability value.
        """
        return trace["logp"]

    def estimate_distribution(self, trace, predictor, **kwargs):
        """
        Estimate the distribution parameters.

        Parameters
        ----------
        trace : dict
            Dictionary containing the trace of parameter values.
        predictor : xarray.DataArray
            Predictor data.
        **kwargs :
            Additional arguments - not used.

        Returns
        -------
        distribution
            Estimated distribution.
        """
        params = {}
        for name, parameter_model in self._parameter_models.items():
            parameter_model.set_predictor_data(predictor)
            params[name], _ = parameter_model.estimate(trace["params"])

        return self._distribution_class(**params)
