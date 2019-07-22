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

def det_dot2(x, beta):
    # FIXME: can this be replaced through det_dot?
    return np.dot(x, beta.T)


class Normal(object):

    """ Influence of GMT is modelled through a shift of
    mu (the mean) of a normally distributed variable. """

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

        self.vars_to_estimate = ["slope", "intercept", "sigma", "beta_yearly",
        "beta_trend"]

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
            # mu varies with a yearly cycle, long-term GMT-driven trend and a
            # GMT-driven cycle
            mu = (
                intercept
                + det_dot(x_fourier, beta_yearly)
                + regressor * (slope + det_dot(x_fourier, beta_trend))
            )
            out = pm.Normal("obs", mu=mu, sd=sigma, observed=observed)

        return model

    def quantile_mapping(self, trace, regressor, x_fourier, x):

        """
        specific for normally distributed variables where
        we diagnose shift in mu through GMT.
        """

        # only the GMT-driven parts of mu variability
        # so without the yearly cycle
        mu_gmt = (
            trace["intercept"]
            + regressor[:, None]
            * (
                trace["slope"]
                + det_dot2(x_fourier, trace["beta_trend"])
            )
        ).mean(axis=1)

        mu_reference = mu_gmt[0 : self.reference_time].mean()
        sigma = trace["sigma"].mean()

        quantile = stats.norm.cdf(x, loc=mu_gmt, scale=sigma)
        x_mapped = stats.norm.ppf(
            quantile, loc=mu_reference, scale=sigma
        )

        return x_mapped


class Beta(object):

    """ Influence of GMT is modelled through the influence of on the alpha parameter
    of a Beta distribution. Beta parameter is assumed free of a trend. """

    def setup(self):

        model = pm.Model()

        with model:
            #     slope = pm.Normal("slope", linear_mu, linear_sigma)
            #     intercept = pm.Normal("intercept", linear_mu, linear_sigma)

            #     alpha = pm.Uniform("alpha", 1.5, 20.)
            beta = pm.Uniform("beta", 1.5, 20.0)

            slope = pm.Uniform("slope", -5, 5)
            intercept = pm.Uniform("intercept", 1.5, 15)
            #     sigma = pm.HalfCauchy("sigma", sigma_beta, testval=1)

            beta_yearly = pm.Normal("beta_yearly", mu=smu, sd=sps, shape=2 * modes)
            beta_trend = pm.Normal("beta_trend", mu=stmu, sd=stps, shape=2 * modes)

            param_gmt = (
                intercept
                + slope * regressor
                + det_dot(x_fourier, beta_yearly)
                + (regressor * det_dot(x_fourier, beta_trend))
            )
            #     pr_intensity = pm.Gamma('pr_intensity', alpha=alpha, beta=beta,
            #                    observed=tdf_valid["pr_intensity"])
            out = pm.Beta("obs", alpha=param_gmt, beta=beta, observed=tdf["y_scaled"])
