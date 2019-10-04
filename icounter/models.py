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


def get_mu_sigma(
    trace,
    regressor,
    x_fourier,
    date_index,
    nd
):

    """ TODO: rename variables in here. """

    # FIXME: bring this to settings.py
    ref_start_date = "1901-01-01"
    ref_end_date = "1910-12-31"

    param_gmt = (
        trace[nd["intercept"]]
        + np.dot(x_fourier, trace[nd["yearly"]].T)
        + regressor[:, None]
        * (trace[nd["slope"]] + np.dot(x_fourier, trace[nd["trend"]].T))
    ).mean(axis=1)

    df_param = pd.DataFrame({"param_gmt": param_gmt}, index=date_index)
    # restrict data to the reference period
    df_param_ref = df_param.loc[ref_start_date:ref_end_date]
    # mean over all years for each day
    df_param_ref = df_param_ref.groupby(df_param_ref.index.dayofyear).mean()

    # write the average values for the reference period to each day of the
    # whole timeseries
    for day in df_param_ref.index:
        df_param.loc[
            df_param.index.dayofyear == day, "param_gmt_ref"
        ] = df_param_ref.loc[day].values[0]

    return df_param


def get_sigma_parameter(trace, regressor, date_index):

    # FIXME: bring this to settings.py
    ref_start_date = "1901-01-01"
    ref_end_date = "1910-12-31"

    param_gmt = (
        trace["sigma_intercept"] + regressor[:, None] * trace["sigma_slope"]
    ).mean(axis=1)

    df_param = pd.DataFrame({"sigma_gmt": param_gmt}, index=date_index)
    # restrict data to the reference period
    df_param_ref = df_param.loc[ref_start_date:ref_end_date]
    # mean over each day in the year
    df_param_ref = df_param_ref.groupby(df_param_ref.index.dayofyear).mean()

    # write the average values for the reference period to each day of the
    # whole timeseries
    for day in df_param_ref.index:
        df_param.loc[
            df_param.index.dayofyear == day, "sigma_gmt_ref"
        ] = df_param_ref.loc[day].values[0]

    return df_param


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

    def __init__(self, modes=1, scale_sigma_with_gmt=True):

        # TODO: allow this to be changed by argument to __init__
        self.modes = modes
        self.scale_sigma_with_gmt = scale_sigma_with_gmt
        # self.mu_intercept = 0.0
        # self.mu_slope = 0
        # self.sigma_intercept = 0.5
        # self.sigma_slope = 1.0
        self.smu = 0
        self.sps = 0.5
        self.stmu = 0
        self.stps = 0.5

        self.vars_to_estimate = [
            "mu_slope",
            "mu_intercept",
            "mu_yearly",
            "mu_trend",
            "sg_slope",
            "sg_intercept",
            "sg_yearly",
            "sg_trend",
        ]

        print("Using Gamma distribution model. Fourier modes:", modes)

    def setup(self, regressor, x_fourier, observed):

        model = pm.Model()

        with model:

            mu_intercept = pm.Lognormal("mu_intercept", mu=0, sigma=1.)
            mu_slope = pm.Normal("mu_slope", mu=0, sigma=2.0)
            mu_yearly = pm.Normal(
                "mu_yearly", mu=0.0, sd=5.0, shape=2 * self.modes
            )
            mu_trend = pm.Normal(
                "mu_trend", mu=0.0, sd=2.0, shape=2 * self.modes
            )
            mu = mu_intercept/(1+tt.exp(-1*(
                mu_slope * regressor
                + det_dot(x_fourier, mu_yearly)
                + regressor * det_dot(x_fourier, mu_trend)))
            )

            sg_intercept = pm.Lognormal("sg_intercept", mu=0, sigma=1.)
            sg_slope = pm.Normal("sg_slope", mu=0, sigma=1)
            sg_yearly = pm.Normal(
                "sg_yearly", mu=0.0, sd=5.0, shape=2 * self.modes
            )
            sg_trend = pm.Normal(
                "sg_trend", mu=0.0, sd=2.0, shape=2 * self.modes
            )
            sigma = sg_intercept/(1+tt.exp(-1*(
                sg_slope * regressor
                + det_dot(x_fourier, sg_yearly)
                + regressor * det_dot(x_fourier, sg_trend)))
            )

            pm.Gamma("obs", mu=mu, sigma=sigma, observed=observed)
            return model

    def quantile_mapping(self, trace, regressor, x_fourier, date_index, x):

        """
        specific for Gamma distributed variables where
        we diagnose shift in beta parameter through GMT.
        """

        df_mu = get_mu_sigma(trace, regressor, x_fourier, date_index, nd={
                "intercept": "mu_intercept",
                "slope": "mu_slope",
                "yearly": "mu_yearly",
                "trend": "mu_trend",
            })

        df_sigma = get_mu_sigma(
            trace,
            regressor,
            x_fourier,
            date_index,
            nd={
                "intercept": "sg_intercept",
                "slope": "sg_slope",
                "yearly": "sg_yearly",
                "trend": "sg_trend",
            },
        )

        mu = df_mu["param_gmt"]
        mu_ref = df_mu["param_gmt_ref"]
        sigma = df_sigma["param_gmt"]
        sigma_ref = sigma
        if self.scale_sigma_with_gmt:
            sigma_ref = df_sigma["param_gmt_ref"]

        # scipy gamma works with alpha and scale parameter
        # alpha=mu**2/sigma**2, scale=1/beta=sigma**2/mu
        quantile = stats.gamma.cdf(x, mu ** 2.0 / sigma ** 2.0, scale=sigma ** 2.0 / mu)
        x_mapped = stats.gamma.ppf(
            quantile, mu_ref ** 2.0 / sigma_ref ** 2.0, scale=sigma_ref ** 2.0 / mu_ref
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
