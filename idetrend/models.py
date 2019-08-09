import numpy as np
import pymc3 as pm
from scipy import stats


def det_dot(a, b):
    """
    The theano dot product and NUTS sampler don't work with large matrices?

    :param a: (np matrix)
    :param b: (theano vector)
    """
    return (a * b[None, :]).sum(axis=-1)


class Normal(object):

    """ Influence of GMT is modelled through a shift of
    mu (the mean) of a normally distributed variable. Works for example for tas."""

    def __init__(self):

        # TODO: allow this to be changed by argument to __init__
        self.modes = 3
        self.linear_mu = 1
        self.linear_sigma = 5
        self.sigma_beta = 0.5
        self.smu = 1
        self.sps = 20
        self.stmu = 0.5
        self.stps = 20

        # reference for quantile mapping
        self.reference_time = 5 * 365

        self.vars_to_estimate = [
            "slope",
            "intercept",
            "sigma",
            "beta_yearly",
            "beta_trend",
        ]

        print("Using Normal distribution model.")

    def setup(self, regressor, x_fourier, observed):

        model = pm.Model()

        with model:
            slope = pm.Normal("slope", self.linear_mu, self.linear_sigma)
            intercept = pm.Normal("intercept", self.linear_mu, self.linear_sigma)
            sigma = pm.HalfCauchy("sigma", self.sigma_beta, testval=1)

            beta_yearly = pm.Normal(
                "beta_yearly", mu=self.smu, sd=self.sps, shape=2 * self.modes
            )
            beta_trend = pm.Normal(
                "beta_trend", mu=self.stmu, sd=self.stps, shape=2 * self.modes
            )
            mu = (
                intercept
                + det_dot(x_fourier, beta_yearly)
                + regressor * (slope + det_dot(x_fourier, beta_trend))
            )
            pm.Normal("obs", mu=mu, sd=sigma, observed=observed)

        return model

    def quantile_mapping(self, trace, regressor, x_fourier, x):

        """
        specific for normally distributed variables where
        we diagnose shift in mu through GMT.
        """

        mu_gmt = (
            trace["intercept"]
            + regressor[:, None]
            * (trace["slope"] + np.dot(x_fourier, trace["beta_trend"].T))
        ).mean(axis=1)

        mu_reference = mu_gmt[0 : self.reference_time].mean()
        sigma = trace["sigma"].mean()

        quantile = stats.norm.cdf(x, loc=mu_gmt, scale=sigma)
        x_mapped = stats.norm.ppf(quantile, loc=mu_reference, scale=sigma)

        return x_mapped


class Gamma(object):

    """ Influence of GMT is modelled through the influence of on the alpha parameter
    of a Beta distribution. Beta parameter is assumed free of a trend.
    Example: precipitation """

    def __init__(self, modes=3):

        # TODO: allow this to be changed by argument to __init__
        self.modes = modes
        self.linear_mu = 1
        self.linear_sigma = 5
        self.smu = 0
        self.sps = 0.1
        self.stmu = 0
        self.stps = 0.1

        # reference for quantile mapping
        self.reference_time = 5 * 365

        self.vars_to_estimate = [
            "slope",
            "intercept",
            "alpha",
            "beta_yearly",
            "beta_trend",
        ]

        print("Using gamma distribution model.")


    def setup(self, regressor, x_fourier, observed):

        model = pm.Model()

        with model:
            alpha = pm.Uniform("alpha", 0.01, 1)
            slope = pm.Uniform("slope", -10, 10)
            intercept = pm.Uniform("intercept", -1, 100)

            beta_yearly = pm.Normal(
                "beta_yearly", mu=self.smu, sd=self.sps, shape=2 * self.modes
            )
            beta_trend = pm.Normal(
                "beta_trend", mu=self.stmu, sd=self.stps, shape=2 * self.modes
            )

            beta = (
                intercept
                + slope * regressor
                + det_dot(x_fourier, beta_yearly)
                + (regressor * det_dot(x_fourier, beta_trend))
            )
            pm.Gamma("obs", alpha=alpha, beta=beta, observed=observed)
            return model

    def quantile_mapping(self, trace, regressor, x_fourier, x):

        """
        specific for Gamma distributed variables where
        we diagnose shift in beta parameter through GMT.
        """
        gmt_driven_trend = (
            regressor[:, None]
            * (trace["slope"] + np.dot(x_fourier, trace["beta_trend"].T))
        ).mean(axis=1)

        alpha = trace["alpha"].mean()
        beta = gmt_driven_trend + trace["intercept"].mean()
        beta_reference = beta[0 : self.reference_time].mean()

        quantile = stats.gamma.cdf(x, trace["alpha"].mean(), scale=1.0 / beta)
        x_mapped = stats.gamma.ppf(quantile, alpha, scale=1.0 / beta_reference)

        return x_mapped


class Beta(object):

    """ Influence of GMT is modelled through the influence of on the alpha parameter
    of a Beta distribution. Beta parameter is assumed free of a trend. """

    def __init__(self):

        # TODO: allow this to be changed by argument to __init__
        self.modes = 3
        self.linear_mu = 0
        self.linear_sigma = 5
        self.sigma_beta = 0.5
        self.smu = 0
        self.sps = 20
        self.stmu = 0
        self.stps = 20

        # reference for quantile mapping
        self.reference_time = 5 * 365

        self.vars_to_estimate = [
            "slope",
            "intercept",
            "beta",
            "beta_yearly",
            "beta_trend",
        ]


        print("Using Gamma distribution model.")


    def setup(self, regressor, x_fourier, observed):

        model = pm.Model()

        with model:
            slope = pm.Uniform("slope", -10, 10)
            intercept = pm.Uniform("intercept", 0.01, 15)
            beta = pm.Uniform("beta", 0.1, 20.0)

            beta_yearly = pm.Normal(
                "beta_yearly", mu=self.smu, sd=self.sps, shape=2 * self.modes
            )
            beta_trend = pm.Normal(
                "beta_trend", mu=self.stmu, sd=self.stps, shape=2 * self.modes
            )

            param_gmt = (
                intercept
                + slope * regressor
                + det_dot(x_fourier, beta_yearly)
                + (regressor * det_dot(x_fourier, beta_trend))
            )

            pm.Beta("obs", alpha=param_gmt, beta=beta, observed=observed)

            return model

    def quantile_mapping(self, trace, regressor, x_fourier, x):

        """
        specific for variables with two bounds, approximately following a
        beta distribution.
        """
        gmt_driven_trend = (
            regressor[:, None]
            * (trace["slope"] + np.dot(x_fourier, trace["beta_trend"].T))
        ).mean(axis=1)

        beta_gmt = gmt_driven_trend + trace["intercept"].mean()
        beta = trace["beta"].mean()
        beta_reference = beta_gmt[0 : self.reference_time].mean()

        quantile = stats.beta.cdf(x, beta_gmt, beta)
        x_mapped = stats.beta.ppf(quantile, beta_reference, beta)

        return x_mapped


class Weibull(object):

    """ Influence of GMT is modelled through the influence of on the shape (alpha) parameter
    of a Weibull distribution. Beta parameter is assumed free of a trend. """

    def __init__(self):

        # TODO: allow this to be changed by argument to __init__
        self.modes = 3
        self.linear_mu = 0
        self.linear_sigma = 5
        self.smu = 0
        self.sps = 5
        self.stmu = 0
        self.stps = 5

        # reference for quantile mapping
        self.reference_time = 5 * 365

        self.vars_to_estimate = [
            "slope",
            "intercept",
            "beta",
            "beta_yearly",
            "beta_trend",
        ]


        print("Using Weibull distribution model.")


    def setup(self, regressor, x_fourier, observed):

        model = pm.Model()

        with model:
            slope = pm.Uniform("slope", -5, 5)
            intercept = pm.Uniform("intercept", -5, 5)
            beta = pm.Uniform("beta", 0.001, 1.0)

            beta_yearly = pm.Normal(
                "beta_yearly", mu=self.smu, sd=self.sps, shape=2 * self.modes
            )
            beta_trend = pm.Normal(
                "beta_trend", mu=self.stmu, sd=self.stps, shape=2 * self.modes
            )

            param_gmt = (
                intercept
                + slope * regressor
                + det_dot(x_fourier, beta_yearly)
                + (regressor * det_dot(x_fourier, beta_trend))
            )

            pm.Weibull("obs", alpha=param_gmt, beta=beta, observed=observed)

            return model

    def quantile_mapping(self, trace, regressor, x_fourier, x):

        """
        specific for variables with two bounds, approximately following a
        beta distribution.
        """
        gmt_driven_trend = (
            regressor[:, None]
            * (trace["slope"] + np.dot(x_fourier, trace["beta_trend"].T))
        ).mean(axis=1)

        beta_gmt = gmt_driven_trend + trace["intercept"].mean()
        beta_reference = beta_gmt[0 : self.reference_time].mean()

        # TODO: Is frechet_r (same as weibull_min, in scipy.stats) really the same as Weibull in pymc3?
        quantile = stats.frechet_r.cdf(x, beta_gmt)
        x_mapped = stats.frechet_r.ppf(quantile, beta_reference)

        return x_mapped


class Rice(object):

    """ Influence of GMT is modelled through shift in the non-concentrality (nu) parameter
    of a Rice distribution. This is useful for normally distributed variables with a lower boundary ot x=0. Sigma parameter is assumed free of a trend. """

    def __init__(self):

        # TODO: allow this to be changed by argument to __init__
        self.modes = 3
        self.linear_mu = 0.1
        self.linear_sigma = 0.2
        self.smu = 0.01
        self.sps = 0.1
        self.stmu = 0.01
        self.stps = 0.1

        # reference for quantile mapping
        self.reference_time = 5 * 365

        self.vars_to_estimate = [
            "slope",
            "intercept",
            "sigma",
            "beta_yearly",
            "beta_trend",
        ]

    def setup(self, regressor, x_fourier, observed):

        model = pm.Model()

        with model:
            slope = pm.Uniform("slope", -5, 5)
            intercept = pm.Uniform("intercept", 1.5, 15)

            beta_yearly = pm.Normal(
                "beta_yearly", mu=self.smu, sd=self.sps, shape=2 * self.modes
            )
            beta_trend = pm.Normal(
                "beta_trend", mu=self.stmu, sd=self.stps, shape=2 * self.modes
            )

            param_gmt = (
                intercept
                + slope * regressor
                + det_dot(x_fourier, beta_yearly)
                + (regressor * det_dot(x_fourier, beta_trend))
            )

            pm.Rice("obs", nu=param_gmt, observed=observed)

            return model

    def quantile_mapping(self, trace, regressor, x_fourier, x):

        """
        specific for variables with two bounds, approximately following a
        beta distribution.
        """
        gmt_driven_trend = (
            regressor[:, None]
            * (trace["slope"] + np.dot(x_fourier, trace["beta_trend"].T))
        ).mean(axis=1)

        beta_gmt = gmt_driven_trend + trace["intercept"].mean()
        beta_reference = beta_gmt[0 : self.reference_time].mean()

        quantile = stats.rice.cdf(x, beta_gmt)
        x_mapped = stats.rice.ppf(quantile, beta_reference)

        return x_mapped
