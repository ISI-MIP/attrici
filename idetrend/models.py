import numpy as np
import pymc3 as pm
from scipy import stats


# TODO: Do we need these functions at all? I dont see, where this saves code
def det_dot(a, b):
    """
    The theano dot product and NUTS sampler don't work with large matrices?

    :param a: (np matrix)
    :param b: (theano vector)
    """
    # TODO: sum() method could be run on output (outside func), so func can be used more generally.
    return (a * b[None, :]).sum(axis=-1)

#  function is not really needed.
# .T object is, is a transposed version of the "beta" matrix
#  def det_dot2(x, beta):
#      # FIXME: can this be replaced through det_dot?
#      return np.dot(x, beta.T)


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
            pm.Normal("obs", mu=mu, sd=sigma, observed=observed)

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
            * (trace["slope"] + np.dot(x_fourier, trace["beta_trend"].T))
        ).mean(axis=1)

        mu_reference = mu_gmt[0 : self.reference_time].mean()
        sigma = trace["sigma"].mean()

        quantile = stats.norm.cdf(x["y_scaled"], loc=mu_gmt, scale=sigma)
        x_mapped = stats.norm.ppf(quantile, loc=mu_reference, scale=sigma)

        return x_mapped


class Beta(object):

    """ Influence of GMT is modelled through the influence of on the alpha parameter
    of a Beta distribution. Beta parameter is assumed free of a trend. """

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
            "beta",
            "beta_yearly",
            "beta_trend",
        ]

    def setup(self, regressor, x_fourier, observed):

        model = pm.Model()

        with model:
            #     slope = pm.Normal("slope", linear_mu, linear_sigma)
            #     intercept = pm.Normal("intercept", linear_mu, linear_sigma)

            #     alpha = pm.Uniform("alpha", 1.5, 20.)
            #  beta = pm.Uniform("beta", 1.5, 20.0)

            slope = pm.Uniform("slope", -5, 5)
            intercept = pm.Uniform("intercept", 1.5, 15)
            #  sigma = pm.HalfCauchy("sigma", sigma_beta, testval=1)

            beta_yearly = pm.Normal("beta_yearly", mu=self.smu, sd=self.sps, shape=2 * self.modes)
            beta_trend = pm.Normal("beta_trend", mu=self.stmu, sd=self.stps, shape=2 * self.modes)

            param_gmt = (
                intercept
                + slope * regressor
                + det_dot(x_fourier, beta_yearly)
                + (regressor * det_dot(x_fourier, beta_trend))
            )
            sigma = pm.Uniform("sigma", 0.001, np.sqrt(param_gmt * (1 - param_gmt)))
            #     pr_intensity = pm.Gamma('pr_intensity', alpha=alpha, beta=beta,
            #                    observed=tdf_valid["pr_intensity"])
            #  pm.Beta("obs", alpha=param_gmt, beta=beta, observed=observed["y_scaled"])
            pm.Beta("obs", mu=param_gmt, sd=sigma, observed=observed["y_scaled"])

            return model

class Gamma(object):

    """ Influence of GMT is modelled through the influence of on the alpha parameter
    of a Beta distribution. Beta parameter is assumed free of a trend. """

    def __init__(self):

        # TODO: allow this to be changed by argument to __init__
        self.modes = 3
        self.linear_mu = 1
        self.linear_sigma = 5
        self.sigma_beta = 0.5
        self.smu = 1
        self.sps = 2
        self.stmu = 0.5
        self.stps = 2
        self.chains = 2

        # reference for quantile mapping
        self.reference_time = 5 * 365

        self.vars_to_estimate = [
            "slope",
            "intercept",
            "alpha",
            "beta_yearly",
            "beta_trend",
        ]

    def setup(self, regressor, x_fourier, observed):

        model = pm.Model()

        with model:
            #  slope = pm.Normal("slope", self.linear_mu, self.linear_sigma)
            #  intercept = pm.Normal("intercept", self.linear_mu, self.linear_sigma)
            alpha = pm.Uniform("alpha", 0.01, 1)
            slope = pm.Uniform("slope", -5, 5)
            intercept = pm.Uniform("intercept", 6, 100)
            #  sigma = pm.HalfCauchy("sigma", self.sigma_beta, testval=1)

            beta_yearly = pm.Normal("beta_yearly", mu=self.smu, sd=self.sps, shape=2 * self.modes)
            beta_trend = pm.Normal("beta_trend", mu=self.stmu, sd=self.stps, shape=2 * self.modes)

            beta = (
                intercept
                + slope * regressor
                + det_dot(x_fourier, beta_yearly)
                + (regressor * det_dot(x_fourier, beta_trend))
            )
            pm.Gamma('pr_intensity', alpha=alpha, beta=beta,
                           observed=observed)
            return model

    def quantile_mapping(self, trace, regressor, x_fourier, x):

        """
        specific for normally distributed variables where
        we diagnose shift in mu through GMT.
        """
        gmt_driven_trend = (
            regressor[:, None]
            * (
                trace["slope"]
                + np.dot(x_fourier, trace["beta_trend"].T)
            )
        ).mean(axis=1)

        # only the GMT-driven parts of mu variability
        # so without the yearly cycle
        beta = gmt_driven_trend + trace["intercept"].mean()
        alpha = trace["alpha"].mean()

        beta_reference = beta[0:self.reference_time].mean()
        #  sigma = trace["sigma"].mean()

        quantile = stats.gamma.cdf(x["y_scaled"], trace["alpha"].mean(),scale=1./beta)
        x_mapped = stats.gamma.ppf(quantile, alpha ,scale=1./beta_reference)

        return x_mapped


