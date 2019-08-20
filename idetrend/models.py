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
        self.mu_intercept = .5
        self.sigma_intercept = 1
        self.mu_slope = 0.
        self.sigma_slope = 1
        self.sigma = 0.5
        self.smu = 0
        self.sps = 2
        self.stmu = 0
        self.stps = 2

        # reference for quantile mapping
        # FIXME: move to settings
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
            slope = pm.Normal("slope", mu=self.mu_slope, sigma=self.sigma_slope)
            intercept = pm.Normal("intercept", mu=self.mu_intercept, sigma=self.sigma_intercept)
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

        # FIXME: bring this to settings.py
        ref_start_date = "1901-01-01"
        ref_end_date = "1910-12-31"

        log_param_gmt = (
            trace["intercept"]
            + np.dot(x_fourier, trace["beta_yearly"].T)
            + regressor[:, None]
            * (trace["slope"] + np.dot(x_fourier, trace["beta_trend"].T))
        ).mean(axis=1)

        df_param = pd.DataFrame({"param_gmt":log_param_gmt},index=date_index)
        # restrict data to the reference period
        df_param_ref = df_param.loc[ref_start_date:ref_end_date]
        # mean over each day in the year
        df_param_ref = df_param_ref.groupby(df_param_ref.index.dayofyear).mean()

        # write the average values for the reference period to each day of the
        # whole timeseries
        for day in df_param_ref.index:
            df_param.loc[df_param.index.dayofyear == day,
                         "param_gmt_ref"] = df_param_ref.loc[day].values[0]

        sigma = trace["sigma"].mean()

        quantile = stats.norm.cdf(x, loc=df_param["param_gmt"], scale=sigma)
        x_mapped = stats.norm.ppf(quantile, loc=df_param["param_gmt_ref"], scale=sigma)

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

    def __init__(self, modes=3):

        # TODO: allow this to be changed by argument to __init__
        self.modes = modes
        self.mu_intercept = 0.5
        self.sigma_intercept = 1.
        self.mu_slope = 0.
        self.sigma_slope = 1.
        self.smu = 0
        self.sps = 1.
        self.stmu = 0
        self.stps = 1.

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
            intercept = pm.Normal("intercept", mu=self.mu_intercept, sigma=self.sigma_intercept)
            beta = pm.Uniform("beta", 0.1, 20.0)

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

            pm.Beta("obs", alpha=log_param_gmt,
                beta=beta, observed=observed)

            return model

    def quantile_mapping(self, trace, regressor, x_fourier, date_index, x):

        """
        specific for variables with two bounds, approximately following a
        beta distribution. Mapping done for each day.
        """

        # FIXME: bring this to settings.py
        ref_start_date = "1901-01-01"
        ref_end_date = "1910-12-31"

        log_param_gmt = (
            trace["intercept"]
            + np.dot(x_fourier, trace["beta_yearly"].T)
            + regressor[:, None]
            * (trace["slope"] + np.dot(x_fourier, trace["beta_trend"].T))
        ).mean(axis=1)

        df_param = pd.DataFrame({"log_param_gmt":log_param_gmt},index=date_index)
        # restrict data to the reference period
        df_param_ref = df_param.loc[ref_start_date:ref_end_date]
        # mean over each day in the year
        df_param_ref = df_param_ref.groupby(df_param_ref.index.dayofyear).mean()

        # write the average values for the reference period to each day of the
        # whole timeseries
        for day in df_param_ref.index:
            df_param.loc[df_param.index.dayofyear == day,
                         "log_param_gmt_ref"] = df_param_ref.loc[day].values[0]

        beta = trace["beta"].mean()

        quantile = stats.beta.cdf(x, df_param["log_param_gmt"], beta)
        x_mapped = stats.beta.ppf(quantile, df_param["log_param_gmt_ref"], beta)

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
    of a Rice distribution.
    This is useful for normally distributed variables with a lower boundary ot x=0.
    Sigma parameter is assumed free of a trend. """

    def __init__(self, modes=3):

        # TODO: allow this to be changed by argument to __init__
        self.modes = modes
        self.smu = 0.01
        self.sps = 1.
        self.stmu = 0.0
        self.stps = 1.

        # reference for quantile mapping
        self.reference_time = 5 * 365

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

            pm.Rice("obs", nu=param_gmt, sigma=sigma, observed=observed)

            return model

    def quantile_mapping(self, trace, regressor, x_fourier, x):

        """
        specific for variables with two bounds, approximately following a
        Rice distribution.
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
