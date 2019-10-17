import numpy as np
import pymc3 as pm
from scipy import stats
import theano.tensor as tt
import pandas as pd


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

    def __init__(self, modes=3):

        # TODO: allow this to be changed by argument to __init__
        self.modes = modes
        self.mu_intercept = 0.5
        self.sigma_intercept = 1
        self.mu_slope = 0.0
        self.sigma_slope = 1
        self.sigma = 0.5
        self.smu = 0
        self.sps = 2
        self.stmu = 0
        self.stps = 2

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
            slope = pm.Normal("slope", mu=self.mu_slope, sigma=self.sigma_slope)
            intercept = pm.Normal(
                "intercept", mu=self.mu_intercept, sigma=self.sigma_intercept
            )
            sigma = pm.HalfCauchy("sigma", self.sigma, testval=1)

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

    def quantile_mapping(self, trace, regressor, x_fourier, date_index, x):

        """
        specific for normally distributed variables. Mapping done for each day.
        """

        df_param = get_gmt_parameter(trace, regressor, x_fourier, date_index)

        sigma = trace["sigma"].mean()

        quantile = stats.norm.cdf(x, loc=df_param["param_gmt"], scale=sigma)
        x_mapped = stats.norm.ppf(quantile, loc=df_param["param_gmt_ref"], scale=sigma)

        return x_mapped


class Gamma(object):

    """ Influence of GMT is modelled through the influence of on the alpha parameter
    of a Beta distribution. Beta parameter is assumed free of a trend.
    Example: precipitation """

    def __init__(self, modes, scale_variability, sigma_model):

        self.modes = modes
        self.scale_variability = scale_variability
        self.sigma_model = sigma_model
        self.vars_to_estimate = [
            "mu_intercept",
            "mu_slope",
            "mu_yearly",
            "mu_trend",
            "sg_intercept",
            "sg_slope",
            "sg_yearly",
            "sg_trend",
        ]

        print("Using Gamma distribution model. Fourier modes:", modes)

    def setup(self, gmt_valid, x_fourier, observed):

        model = pm.Model()

        with model:

            gmt = pm.Data("gmt", gmt_valid)
            xf0 = pm.Data("xf0", x_fourier[0])
            xf1 = pm.Data("xf1", x_fourier[1])
            xf2 = pm.Data("xf2", x_fourier[2])
            xf3 = pm.Data("xf3", x_fourier[3])
            mu_intercept = pm.Lognormal("mu_intercept", mu=0, sigma=1.0)
            mu_slope = pm.Normal("mu_slope", mu=0, sigma=2.0)
            mu_yearly = pm.Normal("mu_yearly", mu=0.0, sd=5.0, shape=2 * self.modes[0])
            mu_trend = pm.Normal("mu_trend", mu=0.0, sd=2.0, shape=2 * self.modes[1])
            # mu_intercept * logistic(gmt,yearly_cycle), strictly positive
            mu = pm.Deterministic(
                "mu",
                mu_intercept
                / (
                    1
                    + tt.exp(
                        -1
                        * (
                            mu_slope * gmt
                            + det_dot(xf0, mu_yearly)
                            + gmt * det_dot(xf1, mu_trend)
                        )
                    )
                ),
            )
            if self.sigma_model == "full":
                sg_intercept = pm.Lognormal("sg_intercept", mu=0, sigma=1.0)
                sg_slope = pm.Normal("sg_slope", mu=0, sigma=1)
                sg_yearly = pm.Normal(
                    "sg_yearly", mu=0.0, sd=5.0, shape=2 * self.modes[2]
                )
                sg_trend = pm.Normal(
                    "sg_trend", mu=0.0, sd=2.0, shape=2 * self.modes[3]
                )
                # sg_intercept * logistic(gmt,yearly_cycle), strictly positive
                sigma = pm.Deterministic(
                    "sigma",
                    self.pm_sigma0(
                        sg_intercept, sg_slope, gmt, xf2, sg_yearly, xf3, sg_trend
                    ),
                )
            elif self.sigma_model == "no_gmt_trend":
                sg_intercept = pm.Lognormal("sg_intercept", mu=0, sigma=1.0)
                sg_yearly = pm.Normal(
                    "sg_yearly", mu=0.0, sd=5.0, shape=2 * self.modes[2]
                )
                # sg_intercept * logistic(gmt,yearly_cycle), strictly positive
                sigma = pm.Deterministic(
                    "sigma", self.pm_sigma1(sg_intercept, xf2, sg_yearly)
                )
                self.vars_to_estimate.remove("sg_slope")
                self.vars_to_estimate.remove("sg_trend")

            elif self.sigma_model == "no_gmt_cycle_trend":
                sg_intercept = pm.Lognormal("sg_intercept", mu=0, sigma=1.0)
                sg_slope = pm.Normal("sg_slope", mu=0, sigma=1)
                sg_yearly = pm.Normal(
                    "sg_yearly", mu=0.0, sd=5.0, shape=2 * self.modes[2]
                )
                # sg_intercept * logistic(gmt,yearly_cycle), strictly positive
                sigma = pm.Deterministic(
                    "sigma", self.pm_sigma2(sg_intercept, sg_slope, gmt, xf2, sg_yearly)
                )
                self.vars_to_estimate.remove("sg_trend")

            else:
                raise NotImplemented

            pm.Gamma("obs", mu=mu, sigma=sigma, observed=observed)
            return model

    def pm_sigma0(self, sg_intercept, sg_slope, gmt, xf2, sg_yearly, xf3, sg_trend):

        return sg_intercept / (
            1
            + tt.exp(
                -1
                * (
                    sg_slope * gmt
                    + det_dot(xf2, sg_yearly)
                    + gmt * det_dot(xf3, sg_trend)
                )
            )
        )

    def pm_sigma1(self, sg_intercept, xf2, sg_yearly):

        return sg_intercept / (1 + tt.exp(-1 * det_dot(xf2, sg_yearly)))

    def pm_sigma2(self, sg_intercept, sg_slope, gmt, xf2, sg_yearly):

        return sg_intercept / (
            1 + tt.exp(-1 * (sg_slope * gmt + det_dot(xf2, sg_yearly)))
        )

    def quantile_mapping(self, d, y_scaled):

        """
        specific for Gamma distributed variables where
        we diagnose shift in beta parameter through GMT.

        # scipy gamma works with alpha and scale parameter
        # alpha=mu**2/sigma**2, scale=1/beta=sigma**2/mu
        """

        quantile = stats.gamma.cdf(
            y_scaled,
            d["mu"] ** 2.0 / d["sigma"] ** 2.0,
            scale=d["sigma"] ** 2.0 / d["mu"],
        )
        x_mapped = stats.gamma.ppf(
            quantile,
            d["mu_ref"] ** 2.0 / d["sigma_ref"] ** 2.0,
            scale=d["sigma_ref"] ** 2.0 / d["mu_ref"],
        )

        return x_mapped


class Beta(object):

    """ Influence of GMT is modelled through the influence of on the alpha parameter
    of a Beta distribution. Beta parameter is assumed free of a trend. """

    def __init__(self, modes=3):

        # TODO: allow this to be changed by argument to __init__
        self.modes = modes
        self.mu_intercept = 0.5
        self.sigma_intercept = 1.0
        self.mu_slope = 0.0
        self.sigma_slope = 1.0
        self.smu = 0
        self.sps = 1.0
        self.stmu = 0
        self.stps = 1.0

        self.vars_to_estimate = [
            "slope",
            "intercept",
            "beta",
            "beta_yearly",
            "beta_trend",
        ]

        print("Using Beta distribution model.")

    def setup(self, regressor, x_fourier, observed):

        model = pm.Model()

        with model:
            slope = pm.Normal("slope", mu=self.mu_slope, sigma=self.sigma_slope)
            intercept = pm.Normal(
                "intercept", mu=self.mu_intercept, sigma=self.sigma_intercept
            )
            beta = pm.Lognormal("beta", mu=2, sigma=1)

            beta_yearly = pm.Normal(
                "beta_yearly", mu=self.smu, sd=self.sps, shape=2 * self.modes
            )
            beta_trend = pm.Normal(
                "beta_trend", mu=self.stmu, sd=self.stps, shape=2 * self.modes
            )

            log_param_gmt = tt.exp(
                intercept
                + slope * regressor
                + det_dot(x_fourier, beta_yearly)
                + (regressor * det_dot(x_fourier, beta_trend))
            )

            pm.Beta("obs", alpha=log_param_gmt, beta=beta, observed=observed)

            return model

    def quantile_mapping(self, trace, regressor, x_fourier, date_index, x):

        """
        specific for variables with two bounds, approximately following a
        beta distribution. Mapping done for each day.
        """

        df_log = get_gmt_parameter(trace, regressor, x_fourier, date_index)

        beta = trace["beta"].mean()

        quantile = stats.beta.cdf(x, np.exp(df_log["param_gmt"]), beta)
        x_mapped = stats.beta.ppf(quantile, np.exp(df_log["param_gmt_ref"]), beta)

        return x_mapped


class Weibull(object):

    """ Influence of GMT is modelled through the influence of on the shape (alpha) parameter
    of a Weibull distribution. Beta parameter is assumed free of a trend. """

    def __init__(self, modes=3):

        # TODO: allow this to be changed by argument to __init__
        self.modes = modes
        self.mu_intercept = 0.0
        self.sigma_intercept = 1.0
        self.mu_slope = 0.0
        self.sigma_slope = 1.0
        self.smu = 0
        self.sps = 1.0
        self.stmu = 0
        self.stps = 1.0

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
            slope = pm.Normal("slope", mu=self.mu_slope, sigma=self.sigma_slope)
            intercept = pm.Normal(
                "intercept", mu=self.mu_intercept, sigma=self.sigma_intercept
            )
            beta = pm.Lognormal("beta", mu=2, sigma=1)

            beta_yearly = pm.Normal(
                "beta_yearly", mu=self.smu, sd=self.sps, shape=2 * self.modes
            )
            beta_trend = pm.Normal(
                "beta_trend", mu=self.stmu, sd=self.stps, shape=2 * self.modes
            )

            log_param_gmt = tt.exp(
                intercept
                + slope * regressor
                + det_dot(x_fourier, beta_yearly)
                + (regressor * det_dot(x_fourier, beta_trend))
            )

            pm.Weibull("obs", alpha=log_param_gmt, beta=beta, observed=observed)

            return model

    def quantile_mapping(self, trace, regressor, x_fourier, date_index, x):

        """
        specific for variables with two bounds, approximately following a
        Weibull distribution.
        """

        df_log = get_gmt_parameter(trace, regressor, x_fourier, date_index)

        beta = trace["beta"].mean()

        quantile = stats.weibull_min.cdf(x, np.exp(df_log["param_gmt"]), scale=beta)
        x_mapped = stats.weibull_min.ppf(
            quantile, np.exp(df_log["param_gmt_ref"]), scale=beta
        )

        return x_mapped


class Rice(object):

    """ Influence of GMT is modelled through shift in the non-concentrality (nu) parameter
    of a Rice distribution.
    This is useful for normally distributed variables with a lower boundary ot x=0.
    Sigma parameter is assumed free of a trend. """

    def __init__(self, modes=3):

        # TODO: allow this to be changed by argument to __init__
        self.modes = modes
        self.mu_intercept = 0.0
        self.sigma_intercept = 1.0
        self.mu_slope = 0.0
        self.sigma_slope = 1.0
        self.smu = 0
        self.sps = 1.0
        self.stmu = 0
        self.stps = 1.0

        self.vars_to_estimate = [
            "slope",
            "intercept",
            "sigma",
            "beta_yearly",
            "beta_trend",
        ]

        print("Using Rice distribution model.")

    def setup(self, regressor, x_fourier, observed):

        model = pm.Model()

        with model:

            slope = pm.Normal("slope", mu=self.mu_slope, sigma=self.sigma_slope)
            intercept = pm.Normal(
                "intercept", mu=self.mu_intercept, sigma=self.sigma_intercept
            )
            sigma = pm.Lognormal("sigma", mu=2, sigma=1)

            beta_yearly = pm.Normal(
                "beta_yearly", mu=self.smu, sd=self.sps, shape=2 * self.modes
            )
            beta_trend = pm.Normal(
                "beta_trend", mu=self.stmu, sd=self.stps, shape=2 * self.modes
            )

            log_param_gmt = tt.exp(
                intercept
                + slope * regressor
                + det_dot(x_fourier, beta_yearly)
                + (regressor * det_dot(x_fourier, beta_trend))
            )

            pm.Rice("obs", nu=log_param_gmt, sigma=sigma, observed=observed)

            return model

    def quantile_mapping(self, trace, regressor, x_fourier, date_index, x):

        """
        specific for variables with two bounds, approximately following a
        Rice distribution.
        """

        df_log = get_gmt_parameter(trace, regressor, x_fourier, date_index)

        sigma = trace["sigma"].mean()

        quantile = stats.rice.cdf(x, np.exp(df_log["param_gmt"]), scale=sigma)
        x_mapped = stats.rice.ppf(
            quantile, np.exp(df_log["param_gmt_ref"]), scale=sigma
        )

        return x_mapped
