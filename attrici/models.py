import numpy as np
import pymc3 as pm
from scipy import stats
import pandas as pd
import theano.tensor as tt
import attrici.distributions


def det_dot(a, b):
    """
    The theano dot product and NUTS sampler don't work with large matrices?
    :param a: (np matrix)
    :param b: (theano vector)
    """
    return (a * b[None, :]).sum(axis=-1)


def logit(eta):
    return 1 / (1 + tt.exp(-eta))


class PrecipitationLongterm(attrici.distributions.BernoulliGamma):

    """ Influence of GMT is modelled through the parameters of the Gamma
    distribution. Example: precipitation """

    def __init__(self, modes):
        super(PrecipitationLongterm, self).__init__()
        self.modes = modes
        self.test = False

    def setup(self, df_subset):

        model = pm.Model()

        with model:
            # dropna to make sampling possible for the precipitation amounts.
            df_valid = df_subset.dropna(axis=0, how="any")
            gmt = pm.Data("gmt", df_subset["gmt_scaled"].values)
            gmtv = pm.Data("gmtv", df_valid["gmt_scaled"].values)

            # pbern
            # b is in the interval (0,2)
            b = pm.Beta(
                "pbern_b", alpha=2, beta=2
            )  # beta(2, 2) is symmetric with mode at 0.5 b is in
            # a is in the interval (-b,1-b)
            a = tt.sub(pm.Beta("pbern_a", alpha=2, beta=2), b)
            # pbern is in the interval (0,1)
            pbern = a * gmt + b  # pbern is a linear model of gmt
            pbern = pm.Deterministic("pbern", pbern)

            # mu
            # b_mu is in the interval (0,inf)
            b_mu = pm.Exponential("b_mu", lam=1)
            # a_mu in (-b, inf)
            a_mu = pm.Deterministic("a_mu", pm.Exponential("am", lam=1) - b_mu)
            mu = pm.Deterministic("mu", a_mu * gmtv + b_mu)  # in (0, inf)

            # sigma
            sigma = pm.Lognormal("sigma", mu=-1, sigma=0.4, testval=1.0)

            if not self.test:
                pm.Bernoulli(
                    "bernoulli", p=pbern, observed=df_subset["is_dry_day"].astype(int)
                )
                pm.Gamma("obs", mu=mu, sigma=sigma, observed=df_valid["y_scaled"])

        return model


class PrecipitationLongtermTrendSigma(attrici.distributions.BernoulliGamma):

    """ Influence of GMT is modelled through the parameters of the Gamma
    distribution. Example: precipitation """

    def __init__(self, modes):
        super(PrecipitationLongtermTrendSigma, self).__init__()
        self.modes = modes
        self.test = False

    def setup(self, df_subset):

        model = pm.Model()

        with model:
            # dropna to make sampling possible for the precipitation amounts.
            df_valid = df_subset.dropna(axis=0, how="any")
            gmt = pm.Data("gmt", df_subset["gmt_scaled"].values)
            gmtv = pm.Data("gmtv", df_valid["gmt_scaled"].values)

            # pbern
            # b is in the interval (0,2)
            b = pm.Beta(
                "pbern_b", alpha=2, beta=2
            )  # beta(2, 2) is symmetric with mode at 0.5 b is in
            # a is in the interval (-b,1-b)
            a = tt.sub(pm.Beta("pbern_a", alpha=2, beta=2), b)
            # pbern is in the interval (0,1)
            pbern = a * gmt + b  # pbern is a linear model of gmt
            pbern = pm.Deterministic("pbern", pbern)

            # mu
            # b_mu is in the interval (0,inf)
            b_mu = pm.Exponential("b_mu", lam=1)
            # a_mu in (-b, inf)
            a_mu = pm.Deterministic("a_mu", pm.Exponential("am", lam=1) - b_mu)
            mu = pm.Deterministic("mu", a_mu * gmtv + b_mu)  # in (0, inf)

            # sigma
            # should be same for b and a, so that a is symmetric around zero
            lam = 1
            # b_sigma is in the interval (0,inf)
            b_sigma = pm.Exponential("b_sigma", lam=lam)
            # a_sigma is in the interval (-b, inf), mode at 0
            a_sigma = pm.Deterministic(
                "a_sigma", tt.sub(pm.Exponential("as", lam=lam), b_sigma)
            )
            # sigma in (0, inf)
            sigma = pm.Deterministic("sigma", a_sigma * gmtv + b_sigma)

            if not self.test:
                pm.Bernoulli(
                    "bernoulli", p=pbern, observed=df_subset["is_dry_day"].astype(int)
                )
                pm.Gamma("obs", mu=mu, sigma=sigma, observed=df_valid["y_scaled"])

        return model


class PrecipitationLongtermRelu(attrici.distributions.BernoulliGamma):

    """ Influence of GMT is modelled through the parameters of the Gamma
    distribution. Example: precipitation """

    def __init__(self, modes):
        super(PrecipitationLongtermRelu, self).__init__()
        self.modes = modes
        self.test = False

    def setup(self, df_subset):

        model = pm.Model()

        with model:

            # dropna to make sampling possible for the precipitation amounts.
            df_valid = df_subset.dropna(axis=0, how="any")
            gmt = pm.Data("gmt", df_subset["gmt_scaled"].values)
            gmtv = pm.Data("gmtv", df_valid["gmt_scaled"].values)

            # pbern
            # b_pbern = pm.Normal("pbern_b", mu=0, sigma=1)
            b_pbern = pm.Normal("pbern_b", mu=0.5, sigma=0.2)
            # b_pbern = pm.Beta("pbern_b", alpha=2, beta=2)
            a_pbern = pm.Normal("pbern_a", mu=0, sigma=0.2, testval=0)
            # pbern is a linear model of gmt
            pbern_linear = pm.Deterministic("pbern_linear", a_pbern * gmt + b_pbern)
            pbern = pm.Deterministic("pbern", tt.clip(pbern_linear, 0.001, 0.999))
            # pbern = pm.Deterministic("pbern", tt.nnet.hard_sigmoid(pbern_linear))
            # pbern = pm.Beta("pbern", alpha=2, beta=2)

            # mu
            b_mu = pm.Normal("mu_b", mu=1, sigma=0.5, testval=1.0)
            a_mu = pm.Normal("mu_a", mu=0, sigma=0.5, testval=0)
            mu_linear = pm.Deterministic("mu_linear", a_mu * gmtv + b_mu)
            mu = pm.Deterministic("mu", tt.nnet.relu(mu_linear) + 1e-10)

            # sigma
            b_sigma = pm.Normal("sigma_b", mu=1, sigma=0.5, testval=1.0)
            a_sigma = pm.Normal("sigma_a", mu=0, sigma=1, testval=0)
            sigma_linear = pm.Deterministic("sigma_linear", a_sigma * gmtv + b_sigma)
            sigma = pm.Deterministic("sigma", tt.nnet.relu(sigma_linear) + 1e-6)

            if not self.test:
                pm.Bernoulli(
                    "bernoulli", p=pbern, observed=df_subset["is_dry_day"].astype(int)
                )
                pm.Gamma("obs", mu=mu, sigma=sigma, observed=df_valid["y_scaled"])

        return model


class PrecipitationLongtermYearlycycle(attrici.distributions.BernoulliGamma):
    def __init__(self, modes):
        super(PrecipitationLongtermYearlycycle, self).__init__()
        self.modes = modes
        self.test = False

    def setup(self, df_subset):

        model = pm.Model()

        with model:
            df_valid = df_subset.dropna(axis=0, how="any")
            gmt = pm.Data("gmt", df_subset["gmt_scaled"].values)
            xf0 = pm.Data("xf0", df_subset.filter(like="mode_0_").values)

            gmtv = pm.Data("gmtv", df_valid["gmt_scaled"].values)
            xf0v = pm.Data("xf0v", df_valid.filter(like="mode_0_").values)

            # pbern
            # Unclear if we need scaling of b_pbern_yearly
            # Does not provide improved results as compared to longterm,
            # though more than 25x the CPU time
            pbern_fourier_coeffs = pm.Beta(
                "pbern_fourier_coeffs", alpha=2, beta=3, shape=xf0.dshape[1]
            )
            b_const = pm.Beta("pbern_b_const", alpha=2, beta=2)
            b_pbern_yearly = det_dot(xf0, pbern_fourier_coeffs)
            b_scale = (
                pm.Beta("b_scale", alpha=0.5, beta=1.0)
                * (1 - b_const)
                / tt.max(b_pbern_yearly)
            )
            b_pbern = pm.Deterministic("b_pbern", b_const + b_scale * b_pbern_yearly)
            # pp = pm.Deterministic("ttmax",tt.max(b_pbern_yearly))
            # a_pbern is in the interval (-b_pbern,1-b_pbern)
            a_pbern = pm.Deterministic(
                "a_pbern", pm.Beta("pbern_a", alpha=2, beta=2) - b_const
            )
            pbern = pm.Deterministic("pbern", a_pbern * gmt + b_pbern)

            # mu
            # mu_fourier_coeffs is in the range (0,inf)
            mu_fourier_coeffs = pm.Exponential(
                "mu_fourier_coeffs", lam=1, shape=xf0v.dshape[1]
            )
            b_mu_yearly = det_dot(xf0v, mu_fourier_coeffs)
            b_mu_const = pm.Exponential("mu_b_const", lam=1)

            # b_mu is in the interval (0,inf)
            b_mu = pm.Deterministic("b_mu", b_mu_const + b_mu_const)
            # a_mu in (-b_pbern, inf)
            a_mu = pm.Deterministic("a_mu", pm.Exponential("am", lam=1) - b_mu)
            mu = pm.Deterministic("mu", a_mu * gmtv + b_mu)  # in (0, inf)

            # sigma
            # sigma_fourier_coeffs is in the range (0,inf)
            sigma_fourier_coeffs = pm.Exponential(
                "sigma_fourier_coeffs", lam=1, shape=xf0v.dshape[1]
            )
            b_sigma_yearly = det_dot(xf0v, sigma_fourier_coeffs)
            b_sigma_const = pm.Exponential("sigma_b_const", lam=1)

            # b_sigma is in the interval (0,inf)
            b_sigma = pm.Deterministic("b_sigma", b_sigma_const + b_sigma_yearly)
            # a_sigma in (-b_pbern, inf)
            a_sigma = pm.Deterministic(
                "a_sigma", pm.Exponential("as", lam=1) - b_sigma_const
            )
            sigma = pm.Deterministic("sigma", a_sigma * gmtv + b_sigma)  # in (0, inf)

            if not self.test:
                pm.Bernoulli(
                    "bernoulli", p=pbern, observed=df_subset["is_dry_day"].astype(int)
                )
                pm.Gamma("obs", mu=mu, sigma=sigma, observed=df_valid["y_scaled"])

        return model


class TasLongterm(attrici.distributions.Normal):

    """ Influence of GMT is modelled through a shift of
    mu and sigma parameters in a Normal distribution.
    """

    def __init__(self, modes):
        super(TasLongterm, self).__init__()
        self.modes = modes

    def setup(self, df_subset):

        model = pm.Model()

        with model:
            df_valid = df_subset.dropna(axis=0, how="any")
            gmtv = pm.Data("gmt", df_valid["gmt_scaled"].values)

            # b_mu is in the interval (-inf,inf)
            b_mu = pm.Normal("b_mu", mu=0.5, sigma=1)
            # a_mu in (-inf, inf)
            a_mu = pm.Normal("a_mu", mu=0, sigma=1)
            # in (-inf, inf)
            mu = pm.Deterministic("mu", a_mu * gmtv + b_mu)

            # should be same for b and a, so that a is symmetric around zero
            lam = 1
            # b_sigma is in the interval (0,inf)
            b_sigma = pm.Exponential("b_sigma", lam=lam)
            # a_sigma is in the interval (-b, inf), mode at 0
            a_sigma = pm.Deterministic(
                "a_sigma", tt.sub(pm.Exponential("as", lam=lam), b_sigma)
            )
            # sigma in (0, inf)
            sigma = pm.Deterministic("sigma", a_sigma * gmtv + b_sigma)

            pm.Normal("obs", mu=mu, sigma=sigma, observed=df_valid["y_scaled"])

        return model


class TasCycle(attrici.distributions.Normal):

    """ Influence of GMT is modelled through a shift of
    mu and sigma parameters in a Normal distribution.
    """

    def __init__(self, modes):
        super(TasCycle, self).__init__()
        self.modes = modes
        self.test = False

    def setup(self, df_subset):

        model = pm.Model()

        with model:
            # so use df_subset directly.
            df_valid = df_subset.dropna(axis=0, how="any")
            gmtv = pm.Data("gmt", df_valid["gmt_scaled"].values)
            xf0 = pm.Data("xf0", df_valid.filter(like="mode_0_").values)

            # b_mu is in the interval (-inf,inf)
            b_mu = pm.Normal("b_mu", mu=0.5, sigma=1)
            # a_mu in (-inf, inf)
            a_mu = pm.Normal("a_mu", mu=0, sigma=1)

            yearly_mu = pm.Normal("yearly_mu", mu=0.0, sd=1.0, shape=xf0.dshape[1])
            # in (-inf, inf)
            mu = pm.Deterministic("mu", a_mu * gmtv + b_mu + det_dot(xf0, yearly_mu))

            # should be same for b and a, so that a is symmetric around zero
            lam = 1
            # b_sigma is in the interval (0,inf)
            b_sigma = pm.Lognormal("b_sigma", mu=-3, sigma=1)
            yearly_sigma = pm.Lognormal(
                "yearly_sigma", mu=-3, sigma=1, shape=xf0.dshape[1]
            )
            ys = det_dot(xf0, yearly_sigma)
            # a_sigma is in the interval (-b_sigma - ys, inf), mode at 0
            a_sigma = pm.Deterministic(
                "a_sigma", pm.Lognormal("as", mu=-3, sigma=1) - b_sigma
            )

            # sigma in (0, inf)
            sigma = pm.Deterministic("sigma", a_sigma * gmtv + b_sigma + ys)

            if not self.test:
                pm.Normal("obs", mu=mu, sigma=sigma, observed=df_valid["y_scaled"])

        return model


class TasCycleRelu(attrici.distributions.Normal):

    """ Influence of GMT is modelled through a shift of
    mu and sigma parameters in a Normal distribution.
    """

    def __init__(self, modes):
        super(TasCycleRelu, self).__init__()
        self.modes = modes
        self.test = False

    def setup(self, df_subset):

        model = pm.Model()

        with model:
            # so use df_subset directly.
            df_valid = df_subset.dropna(axis=0, how="any")
            gmtv = pm.Data("gmt", df_valid["gmt_scaled"].values)
            xf0 = pm.Data("xf0", df_valid.filter(regex="^mode_0_").values)
            posxf0 = pm.Data("posxf0", df_valid.filter(like="posmode_0_").values)

            # mu
            # b_mu is in the interval (-inf,inf)
            b_mu = pm.Normal("b_mu", mu=0.5, sigma=1)
            # a_mu in (-inf, inf)
            a_mu = pm.Normal("a_mu", mu=0, sigma=1)

            fourier_coefficients_mu = pm.Normal(
                "fourier_coefficients_mu", mu=0.0, sd=1.0, shape=xf0.dshape[1]
            )
            # in (-inf, inf)
            mu = pm.Deterministic(
                "mu", a_mu * gmtv + b_mu + det_dot(xf0, fourier_coefficients_mu)
            )

            # sigma
            b_sigma = pm.Lognormal("b_sigma", mu=-1, sigma=0.4, testval=1.0)
            a_sigma = pm.Normal("a_sigma", mu=0, sigma=0.05, testval=0)

            fourier_coefficients_sigma = pm.Lognormal(
                "fourier_coefficients_sigma", mu=0.0, sd=0.1, shape=posxf0.dshape[1]
            )
            # in (-inf, inf)
            lin = pm.Deterministic(
                "lin",
                a_sigma * gmtv + b_sigma + det_dot(posxf0, fourier_coefficients_sigma),
            )
            alpha = 1e-4
            sigma = pm.Deterministic("sigma", tt.nnet.elu(lin, alpha)) + 2 * alpha
            # sigma = pm.Lognormal("sigma", mu=-1, sigma=1)

            if not self.test:
                pm.Normal("obs", mu=mu, sigma=sigma, observed=df_valid["y_scaled"])

        return model


class TasLogistic(attrici.distributions.Normal):

    """ Influence of GMT is modelled through a shift of
    mu and sigma parameters in a Normal distribution.
    """

    def __init__(self, modes):
        super(TasLogistic, self).__init__()
        self.modes = modes
        self.test = False

    def setup(self, df_subset):

        model = pm.Model()

        with model:
            # so use df_subset directly.
            df_valid = df_subset.dropna(axis=0, how="any")
            gmtv = pm.Data("gmt", df_valid["gmt_scaled"].values)
            xf0 = pm.Data("xf0", df_valid.filter(regex="^mode_0_").values)
            # xf1 = pm.Data("xf1", df_valid.filter(regex="^mode_1_").values)
            xf2 = pm.Data("xf2", df_valid.filter(regex="^mode_2_").values)
            # mu
            # b_mu is in the interval (-inf,inf)
            b_mu = pm.Normal("b_mu", mu=0.5, sigma=1)
            # a_mu in (-inf, inf)
            a_mu = pm.Normal("a_mu", mu=0, sigma=1)

            fourier_coefficients_mu = pm.Normal(
                "fourier_coefficients_mu", mu=0.0, sd=1.0, shape=xf0.dshape[1]
            )
            # in (-inf, inf)
            mu = pm.Deterministic(
                "mu", a_mu * gmtv + b_mu + det_dot(xf0, fourier_coefficients_mu)
            )

            # sigma
            b_sigma = pm.Lognormal("b_sigma", mu=0.0, sigma=1.0)
            a_sigma = pm.Normal("a_sigma", mu=0, sigma=1.0)

            fourier_coeffs_sigma = pm.Lognormal(
                "fourier_coeffs_sigma", mu=0.0, sd=5.0, shape=xf2.dshape[1]
            )
            # in (-inf, inf)
            logistic = b_sigma / (
                1 + tt.exp(-1.0 * (a_sigma * gmtv + det_dot(xf2, fourier_coeffs_sigma)))
            )

            sigma = pm.Deterministic("sigma", logistic)
            # sigma = pm.Lognormal("sigma", mu=-1, sigma=1)

            if not self.test:
                pm.Normal("obs", mu=mu, sigma=sigma, observed=df_valid["y_scaled"])

        return model


class Tas(attrici.distributions.Normal):

    """ Influence of GMT is modelled through a shift of
    mu parameter in the Normal distribution.
    """

    def __init__(self, modes):
        super(Tas, self).__init__()
        self.modes = modes
        self.test = False

    def setup(self, df_subset):

        model = pm.Model()

        with model:
            # so use df_subset directly.
            df_valid = df_subset.dropna(axis=0, how="any")
            gmtv = pm.Data("gmt", df_valid["gmt_scaled"].values)
            xf0 = pm.Data("xf0", df_valid.filter(regex="^mode_0_").values)
            xf1 = pm.Data("xf1", df_valid.filter(regex="^mode_1_").values)

            # mu
            # b_mu is in the interval (-inf,inf)
            b_mu = pm.Normal("b_mu", mu=0.5, sigma=1)
            # a_mu in (-inf, inf)
            a_mu = pm.Normal("a_mu", mu=0, sigma=1)

            fc_mu = pm.Normal("fc_mu", mu=0.0, sd=2.0, shape=xf0.dshape[1])
            fctrend_mu = pm.Normal("fctrend_mu", mu=0.0, sd=2.0, shape=xf1.dshape[1])

            # in (-inf, inf)
            mu = pm.Deterministic(
                "mu",
                a_mu * gmtv
                + b_mu
                + det_dot(xf0, fc_mu)
                + gmtv * det_dot(xf1, fctrend_mu),
            )

            # sigma
            sigma = pm.Lognormal("sigma", mu=-1, sigma=0.4, testval=1.0)

            if not self.test:
                pm.Normal("obs", mu=mu, sigma=sigma, observed=df_valid["y_scaled"])

        return model


class TasCauchySigmaPrior(attrici.distributions.Normal):

    """ Influence of GMT is modelled through a shift of
    mu parameter in the Normal distribution.
    """

    def __init__(self, modes):
        super(TasCauchySigmaPrior, self).__init__()
        self.modes = modes
        self.test = False

    def setup(self, df_subset):

        model = pm.Model()

        with model:
            # so use df_subset directly.
            df_valid = df_subset.dropna(axis=0, how="any")
            gmtv = pm.Data("gmt", df_valid["gmt_scaled"].values)
            xf0 = pm.Data("xf0", df_valid.filter(regex="^mode_0_").values)
            xf1 = pm.Data("xf1", df_valid.filter(regex="^mode_1_").values)

            # mu
            # b_mu is in the interval (-inf,inf)
            b_mu = pm.Normal("b_mu", mu=0.5, sigma=1)
            # a_mu in (-inf, inf)
            a_mu = pm.Normal("a_mu", mu=0, sigma=1)

            fc_mu = pm.Normal("fc_mu", mu=0.0, sd=2.0, shape=xf0.dshape[1])
            fctrend_mu = pm.Normal("fctrend_mu", mu=0.0, sd=2.0, shape=xf1.dshape[1])

            # in (-inf, inf)
            mu = pm.Deterministic(
                "mu",
                a_mu * gmtv
                + b_mu
                + det_dot(xf0, fc_mu)
                + gmtv * det_dot(xf1, fctrend_mu),
            )

            sigma = pm.HalfCauchy("sigma", 0.5, testval=1)

            if not self.test:
                pm.Normal("obs", mu=mu, sigma=sigma, observed=df_valid["y_scaled"])

        return model


class Rlds(attrici.distributions.Normal):

    """ Influence of GMT on longwave downwelling shortwave radiation
    is modelled through a shift of mu parameter in the Normal distribution.
    """

    def __init__(self, modes):
        super(Rlds, self).__init__()
        self.modes = modes
        self.test = False

    def setup(self, df_subset):

        model = pm.Model()

        with model:
            # so use df_subset directly.
            df_valid = df_subset.dropna(axis=0, how="any")
            gmtv = pm.Data("gmt", df_valid["gmt_scaled"].values)
            xf0 = pm.Data("xf0", df_valid.filter(regex="^mode_0_").values)
            xf1 = pm.Data("xf1", df_valid.filter(regex="^mode_1_").values)

            # mu
            # b_mu is in the interval (-inf,inf)
            b_mu = pm.Normal("b_mu", mu=0.5, sigma=1)
            # a_mu in (-inf, inf)
            a_mu = pm.Normal("a_mu", mu=0, sigma=1)

            fc_mu = pm.Normal("fc_mu", mu=0.0, sd=2.0, shape=xf0.dshape[1])
            fctrend_mu = pm.Normal("fctrend_mu", mu=0.0, sd=2.0, shape=xf1.dshape[1])

            # in (-inf, inf)
            mu = pm.Deterministic(
                "mu",
                a_mu * gmtv
                + b_mu
                + det_dot(xf0, fc_mu)
                + gmtv * det_dot(xf1, fctrend_mu),
            )
            # sigma
            sigma = pm.Lognormal("sigma", mu=-1, sigma=0.4, testval=1.0)

            if not self.test:
                pm.Normal("obs", mu=mu, sigma=sigma, observed=df_valid["y_scaled"])

        return model


class RldsConstSigma(attrici.distributions.Normal):

    """ Influence of GMT on longwave downwelling shortwave radiation
    is modelled through a shift of mu parameter in the Normal distribution.
    """

    def __init__(self, modes):
        super(RldsConstSigma, self).__init__()
        self.modes = modes
        self.test = False

    def setup(self, df_subset):

        model = pm.Model()

        with model:
            # so use df_subset directly.
            df_valid = df_subset.dropna(axis=0, how="any")
            gmtv = pm.Data("gmt", df_valid["gmt_scaled"].values)
            xf0 = pm.Data("xf0", df_valid.filter(regex="^mode_0_").values)
            xf1 = pm.Data("xf1", df_valid.filter(regex="^mode_1_").values)

            # mu
            # b_mu is in the interval (-inf,inf)
            b_mu = pm.Normal("b_mu", mu=0.5, sigma=1)
            # a_mu in (-inf, inf)
            a_mu = pm.Normal("a_mu", mu=0, sigma=1)

            fc_mu = pm.Normal("fc_mu", mu=0.0, sd=2.0, shape=xf0.dshape[1])
            fctrend_mu = pm.Normal("fctrend_mu", mu=0.0, sd=2.0, shape=xf1.dshape[1])

            # in (-inf, inf)
            mu = pm.Deterministic(
                "mu",
                a_mu * gmtv
                + b_mu
                + det_dot(xf0, fc_mu)
                + gmtv * det_dot(xf1, fctrend_mu),
            )

            sigma = pm.HalfCauchy("sigma", 0.5, testval=1)

            if not self.test:
                pm.Normal("obs", mu=mu, sigma=sigma, observed=df_valid["y_scaled"])

        return model


class Ps(attrici.distributions.Normal):

    """ Influence of GMT on sea level pressure (ps) is modelled through a shift of
    mu parameter in the Normal distribution.
    """

    def __init__(self, modes):
        super(Ps, self).__init__()
        self.modes = modes
        self.test = False

    def setup(self, df_subset):

        model = pm.Model()

        with model:
            # so use df_subset directly.
            df_valid = df_subset.dropna(axis=0, how="any")
            gmtv = pm.Data("gmt", df_valid["gmt_scaled"].values)
            xf0 = pm.Data("xf0", df_valid.filter(regex="^mode_0_").values)
            xf1 = pm.Data("xf1", df_valid.filter(regex="^mode_1_").values)

            # mu
            # b_mu is in the interval (-inf,inf)
            b_mu = pm.Normal("b_mu", mu=0.5, sigma=1)
            # a_mu in (-inf, inf)
            a_mu = pm.Normal("a_mu", mu=0, sigma=1)

            fc_mu = pm.Normal("fc_mu", mu=0.0, sd=2.0, shape=xf0.dshape[1])
            fctrend_mu = pm.Normal("fctrend_mu", mu=0.0, sd=2.0, shape=xf1.dshape[1])

            # in (-inf, inf)
            mu = pm.Deterministic(
                "mu",
                a_mu * gmtv
                + b_mu
                + det_dot(xf0, fc_mu)
                + gmtv * det_dot(xf1, fctrend_mu),
            )

            sigma = pm.HalfCauchy("sigma", 0.5, testval=1)

            if not self.test:
                pm.Normal("obs", mu=mu, sigma=sigma, observed=df_valid["y_scaled"])

        return model


class HursGLM(attrici.distributions.Beta):
    """ Influence of GMT on relative humidity (hurs) is modelled with Beta regression as proposed in
    https://www.tandfonline.com/doi/abs/10.1080/0266476042000214501
    """

    def __init__(self, modes):
        super(HursGLM, self).__init__()
        self.modes = modes
        self.test = False

    def setup(self, df_subset):
        model = pm.Model()

        with model:
            # The best is to read the model from the bottom up.

            # getting the predictors
            df_valid = df_subset.dropna(axis=0, how="any")
            gmtv = pm.Data("gmt", df_valid["gmt_scaled"].values)
            xf0 = pm.Data("xf0", df_valid.filter(regex="^mode_0_").values)
            # covariates is a [n x d] matrix with all (d-1) predictors it contains the following columns:
            # column of ones which is equivalent to the intercept term in a linear model
            # scaled gmt values
            # 2 columns (sin and cos) for each mode for the yearly cycle that does not depend on GMT
            # 2 columns (sin*GMT and cos*GMT) for each mode for the gmt-dependent change in the yearly cycle
            covariates = pm.math.concatenate(
                [tt.ones_like(gmtv)[:, None],
                 gmtv[:, None],
                 xf0,
                 tt.tile(gmtv[:, None], (1, int(xf0.dshape[1]))) * xf0
                 ],
                axis=1
            )
            # --------------------------------------------------------
            # definition of priors

            # phi is called the precision parameter
            phi = pm.Exponential("phi", lam=0.01)
            # phi = pm.Lognormal("phi", mu=1.0, sigma=1.5, testval=1)

            # mu
            mu_weights = pm.Normal('mu_weights', mu=0., sd=0.1)
            sigma_weights = pm.HalfCauchy('sigma_weights', 1)

            # mu offset is a reparametrization to get a decentralized model,
            # see https://twiecki.io/blog/2017/02/08/bayesian-hierchical-non-centered/
            # and discussion in https://github.com/pymc-devs/pymc3/issues/3132
            mu_weights_offset = pm.Normal('mu_weights_offset', mu=0, sd=10, shape=2 + 2 * xf0.dshape[1])

            # decentralized model
            weights = pm.Deterministic("weights", mu_weights + mu_weights_offset * sigma_weights)
            # centralized model
            # weights = pm.Normal('weights', mu=mu_weights, sd=sigma_weights, shape=2+2*xf0.dshape[1])
            # ----------------------------------------------------------
            # weights are the parameters that are learned by the model
            # eta is a linear model of the predictors
            eta = tt.dot(covariates, weights)
            # mu models the expected value of the target data.
            # The logit function is the link function in the Generalized Linear Model
            mu = pm.math.invlogit(eta)

            # alpha and beta are the parameters for the
            alpha = pm.Deterministic("alpha", mu * phi)

            beta = pm.Deterministic("beta", (1 - mu) * phi)

            if not self.test:
                pm.Beta("obs", alpha=alpha, beta=beta, observed=df_valid["y_scaled"])

        return model


class HursGLM_both_params(attrici.distributions.Beta):
    """ Influence of GMT on relative humidity (hurs) is modelled with Beta regression as proposed in
    https://www.tandfonline.com/doi/abs/10.1080/0266476042000214501
    """

    def __init__(self, modes):
        super(HursGLM_both_params, self).__init__()
        self.modes = modes
        self.test = False

    def setup(self, df_subset):
        model = pm.Model()

        with model:
            # The best is to read the model from the bottom up.

            # getting the predictors
            df_valid = df_subset.dropna(axis=0, how="any")
            gmtv = pm.Data("gmt", df_valid["gmt_scaled"].values)
            xf0 = pm.Data("xf0", df_valid.filter(regex="^mode_0_").values)
            # covariates is a [n x d] matrix with all (d-1) predictors it contains the following columns:
            # column of ones which is equivalent to the intercept term in a linear model
            # scaled gmt values
            # 2 columns (sin and cos) for each mode for the yearly cycle that does not depend on GMT
            # 2 columns (sin*GMT and cos*GMT) for each mode for the gmt-dependent change in the yearly cycle
            covariates = pm.math.concatenate(
                [tt.ones_like(gmtv)[:, None],
                 gmtv[:, None],
                 xf0,
                 tt.tile(gmtv[:, None], (1, int(xf0.dshape[1]))) * xf0
                 ],
                axis=1
            )
            # --------------------------------------------------------
            # definition of priors

            # phi is called the precision parameter
            weights_phi = pm.Normal("weights_phi", mu=0, sd=10, shape=2 + 2 * xf0.dshape[1])
            eta_phi = tt.dot(covariates, weights_phi)
            phi = pm.Deterministic("phi", pm.math.exp(eta_phi))

            # mu

            weights = pm.Normal("weights", mu=0, sd=0.1, shape=2 + 2 * xf0.dshape[1])
            # eta is a linear model of the predictors
            eta = tt.dot(covariates, weights)
            # mu models the expected value of the target data.
            # The logit function is the link function in the Generalized Linear Model
            mu = pm.math.invlogit(eta)

            # alpha and beta are the parameters for the
            alpha = pm.Deterministic("alpha", mu * phi)

            beta = pm.Deterministic("beta", (1 - mu) * phi)

            if not self.test:
                pm.Beta("obs", alpha=alpha, beta=beta, observed=df_valid["y_scaled"])

        return model


class HursGLMInteractiveModification(attrici.distributions.Beta):
    """ Influence of GMT on relative humidity (hurs) is modelled with Beta regression as proposed in
    https://www.tandfonline.com/doi/abs/10.1080/0266476042000214501
    """

    def __init__(self, modes):
        super(HursGLMInteractiveModification, self).__init__()
        self.modes = modes
        self.test = False

    def setup(self, df_subset):
        model = pm.Model()

        with model:
            # The best is to read the model from the bottom up.

            # getting the predictors
            df_valid = df_subset.dropna(axis=0, how="any")
            gmtv = pm.Data("gmt", df_valid["gmt_scaled"].values)
            xf0 = pm.Data("xf0", df_valid.filter(regex="^mode_0_").values)
            # covariates is a [n x d] matrix with all (d-1) predictors it contains the following columns:
            # column of ones which is equivalent to the intercept term in a linear model
            # scaled gmt values
            # 2 columns (sin and cos) for each mode for the yearly cycle that does not depend on GMT
            # 2 columns (sin*GMT and cos*GMT) for each mode for the gmt-dependent change in the yearly cycle
            covariates = pm.math.concatenate(
                [
                    # tt.ones_like(gmtv)[:, None],
                    # gmtv[:, None],
                    xf0,
                    tt.tile(gmtv[:, None], (1, int(xf0.dshape[1]))) * xf0
                ],
                axis=1
            )

            # alpha_mle, beta_mle, _, _ = scipy.stats.beta.fit(df_valid["y_scaled"], floc=0, fscale=1)
            # --------------------------------------------------------
            # definition of priors

            # phi is called the precision parameter

            # phi = pm.Exponential("phi", lam=0.01)
            weights_phi_longterm_intercept = pm.Normal("weights_phi_longterm_intercept", mu=0, sd=1)
            weights_phi_fc_intercept = pm.math.concatenate(
                [pm.Normal(f"weights_phi_fc_intercept_{i}", mu=0, sd=1 / (2 * i + 1), shape=2)
                 for i in range(int(xf0.dshape[1]) // 2)]
            )

            eta_phi = tt.dot(xf0, weights_phi_fc_intercept) + weights_phi_longterm_intercept
            phi = pm.math.exp(eta_phi)
            # weights_phi = pm.Normal("weights_phi",mu=0, sd=10, shape=2 + 2 * xf0.dshape[1])
            # eta_phi = tt.dot(covariates, weights_phi)
            # phi = pm.Deterministic("phi", pm.math.exp(eta_phi))

            # decentralized model
            # mu
            # mu_weights = pm.Normal('mu_weights', mu=0., sd=0.1)
            # sigma_weights = pm.HalfCauchy('sigma_weights', 1)

            # mu offset is a reparametrization to get a decentralized model,
            # see https://twiecki.io/blog/2017/02/08/bayesian-hierchical-non-centered/
            # and discussion in https://github.com/pymc-devs/pymc3/issues/3132
            # mu_weights_offset = pm.Normal('mu_weights_offset', mu=0, sd=10, shape=2 + 2 * xf0.dshape[1])
            # weights = pm.Deterministic("weights", mu_weights + mu_weights_offset * sigma_weights)

            # centralized model
            # weights = pm.Normal('weights', mu=mu_weights, sd=sigma_weights, shape=2+2*xf0.dshape[1])

            # nonhierarchical model (I think)
            weights_longterm_intercept = pm.Normal("weights_longterm_intercept", mu=0, sd=1)
            weights_longterm_trend = pm.Normal("weights_longterm_trend", mu=0, sd=0.1)
            weights_fc_intercept = pm.math.concatenate(
                [pm.Normal(f"weights_fc_intercept_{i}", mu=0, sd=1 / (2 * i + 1), shape=2)
                 for i in range(int(xf0.dshape[1]) // 2)]
            )
            weights_fc_trend = pm.Normal("weights_fc_trend", mu=0, sd=0.1, shape=xf0.dshape[1])
            weights_fc = pm.math.concatenate([
                weights_fc_intercept,
                weights_fc_trend
            ])
            # weights = pm.Normal("weights", mu=0, sd=1, shape=2 + 2 * xf0.dshape[1])

            # ----------------------------------------------------------
            # weights are the parameters that are learned by the model
            # eta is a linear model of the predictors
            eta = tt.dot(covariates, weights_fc) + weights_longterm_intercept + weights_longterm_trend * gmtv

            # mu models the expected value of the target data.
            # The logit function is the link function in the Generalized Linear Model
            mu = pm.math.invlogit(eta)
            phi = pm.math.exp(eta_phi)

            # sigma = pm.Deterministic("sigma", mu * (1 - mu) / (1 + phi))

            # mu = tt.printing.Print('mu')(mu)
            # phi = tt.printing.Print('phi')(phi)

            # alpha and beta are the parameters for the
            alpha = pm.Deterministic("alpha", mu * phi)

            beta = pm.Deterministic("beta", (1 - mu) * phi)
            logp_ = pm.Deterministic("logp", model.logpt)

            if not self.test:
                pm.Beta("obs", alpha=alpha, beta=beta, observed=df_valid["y_scaled"])
                # pm.Beta("obs", mu=mu, sigma=sigma, observed=df_valid["y_scaled"])

        return model


class HursBetaRegression(attrici.distributions.Beta):
    """ Influence of GMT on relative humidity (hurs) is modelled with Beta regression as proposed in
    https://www.tandfonline.com/doi/abs/10.1080/0266476042000214501
    """

    def __init__(self, modes):
        super(HursBetaRegression, self).__init__()
        self.modes = modes
        self.test = False

    def setup(self, df_subset):
        model = pm.Model()

        with model:
            # so use df_subset directly.
            df_valid = df_subset.dropna(axis=0, how="any")
            gmtv = pm.Data("gmt", df_valid["gmt_scaled"].values)
            xf0 = pm.Data("xf0", df_valid.filter(regex="^mode_0_").values)
            xf1 = pm.Data("xf1", df_valid.filter(regex="^mode_1_").values)

            # mu

            # phi
            phi = pm.Exponential("phi", lam=0.5)
            # phi = pm.Lognormal("phi", mu=1.0, sigma=1.5, testval=1)

            b_GMT = pm.Normal("b_GMT", mu=0, sigma=1.0, testval=0)
            a_GMT = pm.Normal("a_GMT", mu=0, sigma=1.0, testval=0)

            b_yearly_cylce = pm.Normal("b_yearly_cylce", mu=0.0, sigma=1.0, shape=xf0.dshape[1])
            a_yearly_cycle = pm.Normal(
                "a_yearly_cycle", mu=0.0, sigma=1.0, shape=xf1.dshape[1]
            )
            eta = b_GMT + a_GMT * gmtv \
                  + det_dot(xf0, b_yearly_cylce) \
                  + gmtv * det_dot(xf1, a_yearly_cycle)
            mu = logit(eta)

            # alpha
            alpha = pm.Deterministic("alpha", mu * phi)

            # beta = pm.HalfCauchy("beta", 0.5, testval=1)
            beta = pm.Deterministic("beta", (1 - mu) * phi)

            if not self.test:
                pm.Beta("obs", alpha=alpha, beta=beta, observed=df_valid["y_scaled"])

        return model


class HursBeta(attrici.distributions.Beta):

    """ Influence of GMT on relative humidity (hurs) is modelled through a shift of
    mu parameter in the Normal distribution.
    """

    def __init__(self, modes):
        super(HursBeta, self).__init__()
        self.modes = modes
        self.test = False

    def setup(self, df_subset):

        model = pm.Model()

        with model:
            # so use df_subset directly.
            df_valid = df_subset.dropna(axis=0, how="any")
            gmtv = pm.Data("gmt", df_valid["gmt_scaled"].values)
            xf0 = pm.Data("xf0", df_valid.filter(regex="^mode_0_").values)
            xf1 = pm.Data("xf1", df_valid.filter(regex="^mode_1_").values)

            # alpha
            b_alpha = pm.Lognormal("b_alpha", mu=1.0, sigma=1.5)
            a_alpha = pm.Normal("a_alpha", mu=0, sigma=1.0)

            fc_alpha = pm.Normal("fc_alpha", mu=0.0, sigma=1.0, shape=xf0.dshape[1])
            fctrend_alpha = pm.Normal(
                "fctrend_alpha", mu=0.0, sigma=1.0, shape=xf1.dshape[1]
            )
            # in (-inf, inf)
            logistic = b_alpha / (
                1
                + tt.exp(
                    -1.0
                    * (
                        a_alpha * gmtv
                        + det_dot(xf0, fc_alpha)
                        + gmtv * det_dot(xf1, fctrend_alpha)
                    )
                )
            )

            alpha = pm.Deterministic("alpha", logistic)

            # beta = pm.HalfCauchy("beta", 0.5, testval=1)
            beta = pm.Lognormal("beta", mu=1.0, sigma=1.5)

            if not self.test:
                pm.Beta("obs", alpha=alpha, beta=beta, observed=df_valid["y_scaled"])

        return model


class Hurs(attrici.distributions.Normal):

    """ Influence of GMT on relative humidity (hurs) is modelled through a shift of
    mu parameter in the Normal distribution.
    """

    def __init__(self, modes):
        super(Hurs, self).__init__()
        self.modes = modes
        self.test = False

    def quantile_mapping(self, d, y_scaled):
        """
         nan values are not quantile-mapped. 100% humidity happens mainly at the poles.
        """

        quantile = stats.norm.cdf(y_scaled, loc=d["mu"], scale=d["sigma"])
        x_mapped = stats.norm.ppf(quantile, loc=d["mu_ref"], scale=d["sigma_ref"])
        np.maximum(x_mapped, 0, x_mapped)
        np.minimum(x_mapped, 1, x_mapped)

        return x_mapped

    def setup(self, df_subset):
        model = pm.Model()

        with model:
            # so use df_subset directly.
            df_valid = df_subset.dropna(axis=0, how="any")
            gmtv = pm.Data("gmt", df_valid["gmt_scaled"].values)
            xf0 = pm.Data("xf0", df_valid.filter(regex="^mode_0_").values)
            xf1 = pm.Data("xf1", df_valid.filter(regex="^mode_1_").values)

            # mu
            b_mu = pm.Lognormal("b_mu", mu=-1, sigma=0.4, testval=1.0)
            a_mu = pm.Normal("a_mu", mu=0, sigma=0.05, testval=0)
            fourier_coefficients_mu = pm.Normal(
                "fourier_coefficients_mu", mu=0.0, sd=0.1, shape=xf0.dshape[1]
            )
            fourier_coefficients_mu_a = pm.Normal(
                "fourier_coefficients_mu_a", mu=0.0, sd=0.1, shape=xf1.dshape[1]
            )
            mu = pm.Deterministic(
                "mu",
                (a_mu + det_dot(xf1, fourier_coefficients_mu_a)) * gmtv + b_mu + det_dot(xf0, fourier_coefficients_mu),
            )

            # sigma
            sigma = pm.Lognormal("sigma", mu=-1, sigma=0.4, testval=1.0)

            if not self.test:
                pm.Normal("obs", mu=mu, sigma=sigma, observed=df_valid["y_scaled"])

        return model


class HursTrendSigma(attrici.distributions.Normal):

    """ Influence of GMT on relative humidity (hurs) is modelled through a shift of
    mu parameter in the Normal distribution.
    """

    def __init__(self, modes):
        super(HursTrendSigma, self).__init__()
        self.modes = modes
        self.test = False

    def quantile_mapping(self, d, y_scaled):
        """
         nan values are not quantile-mapped. 100% humidity happens mainly at the poles.
        """

        quantile = stats.norm.cdf(y_scaled, loc=d["mu"], scale=d["sigma"])
        x_mapped = stats.norm.ppf(quantile, loc=d["mu_ref"], scale=d["sigma_ref"])
        np.maximum(x_mapped, 0, x_mapped)
        np.minimum(x_mapped, 1, x_mapped)

        return x_mapped

    def setup(self, df_subset):
        model = pm.Model()

        with model:
            # so use df_subset directly.
            df_valid = df_subset.dropna(axis=0, how="any")
            gmtv = pm.Data("gmt", df_valid["gmt_scaled"].values)
            xf0 = pm.Data("xf0", df_valid.filter(regex="^mode_0_").values)
            xf1 = pm.Data("xf1", df_valid.filter(regex="^mode_1_").values)

            # mu
            b_mu = pm.Lognormal("b_mu", mu=-1, sigma=0.4, testval=1.0)
            a_mu = pm.Normal("a_mu", mu=0, sigma=0.05, testval=0)
            fourier_coefficients_mu = pm.Normal(
                "fourier_coefficients_mu", mu=0.0, sd=0.1, shape=xf0.dshape[1]
            )
            fourier_coefficients_mu_a = pm.Normal(
                "fourier_coefficients_mu_a", mu=0.0, sd=0.1, shape=xf1.dshape[1]
            )
            mu = pm.Deterministic(
                "mu",
                (a_mu + det_dot(xf1, fourier_coefficients_mu_a)) * gmtv + b_mu + det_dot(xf0, fourier_coefficients_mu),
            )

            # sigma
            b_sigma = pm.Lognormal("b_sigma", mu=-1, sigma=0.4, testval=1.0)
            a_sigma = pm.Normal("a_sigma", mu=0, sigma=0.05, testval=0)

            lin = pm.Deterministic(
                "lin_sigma",
                a_sigma * gmtv + b_sigma
            )
            alpha = 1e-6
            sigma = pm.Deterministic("sigma", pm.math.switch(lin > alpha, lin, alpha))

            if not self.test:
                pm.Normal("obs", mu=mu, sigma=sigma, observed=df_valid["y_scaled"])

        return model


class TasLogisticTrend(attrici.distributions.Normal):

    """ Influence of GMT is modelled through a shift of
    mu and sigma parameters in a Normal distribution.
    """

    def __init__(self, modes):
        super(TasLogisticTrend, self).__init__()
        self.modes = modes
        self.test = False

    def setup(self, df_subset):

        model = pm.Model()

        with model:
            # so use df_subset directly.
            df_valid = df_subset.dropna(axis=0, how="any")
            gmtv = pm.Data("gmt", df_valid["gmt_scaled"].values)
            xf0 = pm.Data("xf0", df_valid.filter(regex="^mode_0_").values)
            xf1 = pm.Data("xf1", df_valid.filter(regex="^mode_1_").values)
            xf2 = pm.Data("xf2", df_valid.filter(regex="^mode_2_").values)
            xf3 = pm.Data("xf3", df_valid.filter(regex="^mode_3_").values)
            # mu
            # b_mu is in the interval (-inf,inf)
            b_mu = pm.Normal("b_mu", mu=0.5, sigma=1)
            # a_mu in (-inf, inf)
            a_mu = pm.Normal("a_mu", mu=0, sigma=1)

            fc_mu = pm.Normal("fc_mu", mu=0.0, sd=1.0, shape=xf0.dshape[1])
            fctrend_mu = pm.Normal("fctrend_mu", mu=0.0, sd=1.0, shape=xf1.dshape[1])
            # in (-inf, inf)
            mu = pm.Deterministic(
                "mu",
                a_mu * gmtv
                + b_mu
                + det_dot(xf0, fc_mu)
                + gmtv * det_dot(xf1, fctrend_mu),
            )

            # sigma
            b_sigma = pm.Lognormal("b_sigma", mu=0.0, sigma=1.0)
            a_sigma = pm.Normal("a_sigma", mu=0, sigma=1.0)

            fc_sigma = pm.Lognormal("fc_sigma", mu=0.0, sd=5.0, shape=xf2.dshape[1])
            fctrend_sigma = pm.Normal(
                "fctrend_sigma", mu=0.0, sd=1.0, shape=xf3.dshape[1]
            )
            # in (-inf, inf)
            logistic = b_sigma / (
                1
                + tt.exp(
                    -1.0
                    * (
                        a_sigma * gmtv
                        + det_dot(xf2, fc_sigma)
                        + gmtv * det_dot(xf3, fctrend_sigma)
                    )
                )
            )

            sigma = pm.Deterministic("sigma", logistic)
            # sigma = pm.Lognormal("sigma", mu=-1, sigma=1)

            if not self.test:
                pm.Normal("obs", mu=mu, sigma=sigma, observed=df_valid["y_scaled"])

        return model


class Tasskew(attrici.distributions.Normal):

    """ Influence of GMT is modelled through a shift of
    mu and sigma parameters in a Beta distribution.
    """

    def __init__(self, modes):
        super(Tasskew, self).__init__()
        self.modes = modes
        self.test = False

    def quantile_mapping(self, d, y_scaled):
        """
         nan values are not quantile-mapped. 100% humidity happens mainly at the poles.
        """

        quantile = stats.norm.cdf(y_scaled, loc=d["mu"], scale=d["sigma"])
        x_mapped = stats.norm.ppf(quantile, loc=d["mu_ref"], scale=d["sigma_ref"])

        x_mapped[x_mapped >= 1] = np.nan
        x_mapped[x_mapped <= 0] = np.nan

        return x_mapped

    def setup(self, df_subset):
        model = pm.Model()

        with model:
            df_valid = df_subset.dropna(axis=0, how="any")
            gmtv = pm.Data("gmt", df_valid["gmt_scaled"].values)
            xf0 = pm.Data("xf0", df_valid.filter(regex="^mode_0_").values)
            xf1 = pm.Data("xf1", df_valid.filter(regex="^mode_1_").values)

            # mu
            b_mu = pm.Lognormal("b_mu", mu=-1, sigma=0.4, testval=1.0)
            a_mu = pm.Normal("a_mu", mu=0, sigma=0.05, testval=0)
            fourier_coefficients_mu = pm.Normal(
                "fourier_coefficients_mu", mu=0.0, sd=0.1, shape=xf0.dshape[1]
            )
            fourier_coefficients_mu_a = pm.Normal(
                "fourier_coefficients_mu_a", mu=0.0, sd=0.1, shape=xf1.dshape[1]
            )
            mu = pm.Deterministic(
                "mu",
                (a_mu + det_dot(xf1, fourier_coefficients_mu_a)) * gmtv + b_mu + det_dot(xf0, fourier_coefficients_mu),
            )

            # sigma
            sigma = pm.Lognormal("sigma", mu=-1, sigma=0.4, testval=1.0)

            if not self.test:
                pm.Normal("obs", mu=mu, sigma=sigma, observed=df_valid["y_scaled"])

        return model


class TasskewSigmaTrend(attrici.distributions.Normal):

    """ Influence of GMT is modelled through a shift of
    mu and sigma parameters in a Beta distribution.
    """

    def __init__(self, modes):
        super(TasskewSigmaTrend, self).__init__()
        self.modes = modes
        self.test = False

    def quantile_mapping(self, d, y_scaled):
        """
         nan values are not quantile-mapped. 100% humidity happens mainly at the poles.
        """

        quantile = stats.norm.cdf(y_scaled, loc=d["mu"], scale=d["sigma"])
        x_mapped = stats.norm.ppf(quantile, loc=d["mu_ref"], scale=d["sigma_ref"])

        x_mapped[x_mapped >= 1] = np.nan
        x_mapped[x_mapped <= 0] = np.nan

        return x_mapped

    def setup(self, df_subset):
        model = pm.Model()

        with model:
            df_valid = df_subset.dropna(axis=0, how="any")
            gmtv = pm.Data("gmt", df_valid["gmt_scaled"].values)
            xf0 = pm.Data("xf0", df_valid.filter(regex="^mode_0_").values)
            xf1 = pm.Data("xf1", df_valid.filter(regex="^mode_1_").values)

            # mu
            b_mu = pm.Lognormal("b_mu", mu=-1, sigma=0.4, testval=1.0)
            a_mu = pm.Normal("a_mu", mu=0, sigma=0.05, testval=0)
            fourier_coefficients_mu = pm.Normal(
                "fourier_coefficients_mu", mu=0.0, sd=0.1, shape=xf0.dshape[1]
            )
            fourier_coefficients_mu_a = pm.Normal(
                "fourier_coefficients_mu_a", mu=0.0, sd=0.1, shape=xf1.dshape[1]
            )
            mu = pm.Deterministic(
                "mu",
                (a_mu + det_dot(xf1, fourier_coefficients_mu_a)) * gmtv + b_mu + det_dot(xf0, fourier_coefficients_mu),
            )

            # sigma
            b_sigma = pm.Lognormal("b_sigma", mu=-1, sigma=0.4, testval=1.0)
            a_sigma = pm.Normal("a_sigma", mu=0, sigma=0.05, testval=0)

            lin = pm.Deterministic(
                "lin_sigma",
                a_sigma * gmtv + b_sigma
            )
            alpha = 1e-6
            sigma = pm.Deterministic("sigma", pm.math.switch(lin > alpha, lin, alpha))

            if not self.test:
                pm.Normal("obs", mu=mu, sigma=sigma, observed=df_valid["y_scaled"])

        return model


class TasskewBeta(attrici.distributions.Beta):

    """ Influence of GMT is modelled through a shift of
    mu and sigma parameters in a Beta distribution.
    """

    def __init__(self, modes):
        super(TasskewBeta, self).__init__()
        self.modes = modes
        self.test = False

    def setup(self, df_subset):

        model = pm.Model()

        with model:
            df_valid = df_subset.dropna(axis=0, how="any")
            gmtv = pm.Data("gmt", df_valid["gmt_scaled"].values)
            xf0 = pm.Data("xf0", df_valid.filter(regex="^mode_0_").values)
            xf1 = pm.Data("xf1", df_valid.filter(regex="^mode_1_").values)

            # alpha
            b_alpha = pm.HalfCauchy("b_alpha", beta=50)
            a_alpha = pm.Normal("a_alpha", mu=0, sigma=1.0)

            fc_alpha = pm.Normal("fc_alpha", mu=0.0, sigma=1.0, shape=xf0.dshape[1])
            fctrend_alpha = pm.Normal(
                "fctrend_alpha", mu=0.0, sigma=1.0, shape=xf1.dshape[1]
            )
            lin_alpha = pm.Deterministic(
                "lin_alpha",
                (a_alpha + det_dot(xf1, fctrend_alpha)) * gmtv + b_alpha +  det_dot(xf0, fc_alpha)
            )
            cutoff = 1e-6
            alpha = pm.Deterministic("alpha", pm.math.switch(lin_alpha > cutoff, lin_alpha, cutoff))

            # beta = pm.HalfCauchy("beta", 0.5, testval=1)
            # beta
            # b_beta = pm.HalfCauchy("b_beta", beta=50)
            # a_beta = pm.Normal("a_beta", mu=0, sigma=1.0)
            #
            # fc_beta = pm.Normal("fc_beta", mu=0.0, sigma=1.0, shape=xf0.dshape[1])
            # fctrend_beta = pm.Normal(
            #     "fctrend_beta", mu=0.0, sigma=1.0, shape=xf1.dshape[1]
            # )
            # lin_beta = pm.Deterministic(
            #     "lin_beta",
            #     (a_beta + det_dot(xf1, fctrend_beta)) * gmtv + b_beta +  det_dot(xf0, fc_beta)
            # )
            # cutoff = 1e-6
            # beta = pm.Deterministic("beta", pm.math.switch(lin_beta > cutoff, lin_beta, cutoff))
            beta = pm.Lognormal("beta", mu=3.0, sigma=1.5)

            if not self.test:
                pm.Beta("obs", alpha=alpha, beta=beta, observed=df_valid["y_scaled"])

        return model


class TasskewBetaLogistic(attrici.distributions.Beta):

    """ Influence of GMT is modelled through a shift of
    mu and sigma parameters in a Beta distribution.
    """

    def __init__(self, modes):
        super(TasskewBetaLogistic, self).__init__()
        self.modes = modes
        self.test = False

    def setup(self, df_subset):

        model = pm.Model()

        with model:
            # so use df_subset directly.
            df_valid = df_subset.dropna(axis=0, how="any")
            gmtv = pm.Data("gmt", df_valid["gmt_scaled"].values)
            xf0 = pm.Data("xf0", df_valid.filter(regex="^mode_0_").values)
            xf1 = pm.Data("xf1", df_valid.filter(regex="^mode_1_").values)

            # alpha
            b_alpha = pm.Lognormal("b_alpha", mu=1.0, sigma=1.5)
            a_alpha = pm.Normal("a_alpha", mu=0, sigma=1.0)

            fc_alpha = pm.Normal("fc_alpha", mu=0.0, sigma=1.0, shape=xf0.dshape[1])
            fctrend_alpha = pm.Normal(
                "fctrend_alpha", mu=0.0, sigma=1.0, shape=xf1.dshape[1]
            )
            # in (-inf, inf)
            logistic = b_alpha / (
                1
                + tt.exp(
                    -1.0
                    * (
                        a_alpha * gmtv
                        + det_dot(xf0, fc_alpha)
                        + gmtv * det_dot(xf1, fctrend_alpha)
                    )
                )
            )

            alpha = pm.Deterministic("alpha", logistic)

            # beta = pm.HalfCauchy("beta", 0.5, testval=1)
            beta = pm.Lognormal("beta", mu=1.0, sigma=1.5)

            if not self.test:
                pm.Beta("obs", alpha=alpha, beta=beta, observed=df_valid["y_scaled"])

        return model


class Rsds(attrici.distributions.Normal):
    """ Influence of GMT is modelled through a shift of
        mu and sigma parameters in a Beta distribution.
        """

    def __init__(self, modes):
        super(Rsds, self).__init__()
        self.modes = modes
        self.test = False


    def quantile_mapping(self, d, y_scaled):
        """
        specific for censored normal values
        rsds can not be smaller than zero
        all values equal to zero are masked out for training
        those masked values are replaced by random values, sampled from the trained model

        those random values are quantile-mapped
        after quantile mapping negative c-fact values are mapped to zero
        """
        random_values = stats.norm.rvs(loc=d["mu"], scale=d["sigma"])
        # the random values should not be larger than 0 (any value smaller 0 is possible)
        np.minimum(random_values, 0, random_values)
        y_filled = y_scaled.fillna(pd.Series(random_values))

        quantile = stats.norm.cdf(y_filled, loc=d["mu"], scale=d["sigma"])
        x_mapped = stats.norm.ppf(quantile, loc=d["mu_ref"], scale=d["sigma_ref"])
        np.maximum(x_mapped, 0, x_mapped)
        return x_mapped

    def setup(self, df_subset):
        model = pm.Model()

        with model:
            # so use df_subset directly.
            df_valid = df_subset.dropna(axis=0, how="any")
            gmtv = pm.Data("gmt", df_valid["gmt_scaled"].values)
            xf0 = pm.Data("xf0", df_valid.filter(regex="^mode_0_").values)
            if (np.array(self.modes) > 1).any():
                raise ValueError('Modes larger 1 are not allowed for the censored model.')

            # mu
            b_mu = pm.Lognormal("b_mu", mu=-1, sigma=0.4, testval=1.0)
            a_mu = pm.Normal("a_mu", mu=0, sigma=0.05, testval=0)
            fourier_coefficients_mu = pm.Normal(
                "fourier_coefficients_mu", mu=0.0, sd=0.1, shape=xf0.dshape[1]
            )
            mu = pm.Deterministic(
                "mu",
                a_mu * gmtv + b_mu + det_dot(xf0, fourier_coefficients_mu),
            )

            # sigma
            sigma = pm.Lognormal("sigma", mu=-1, sigma=0.4, testval=1.0)

            if not self.test:
                pm.Normal("obs", mu=mu, sigma=sigma, observed=df_valid["y_scaled"])

        return model


class RsdsBeta(attrici.distributions.Beta):
    """ Influence of GMT is modelled through a shift of
        mu and sigma parameters in a Beta distribution.
        """

    def __init__(self, modes):
        super(RsdsBeta, self).__init__()
        self.modes = modes
        self.test = False

    def setup(self, df_subset):
        model = pm.Model()

        with model:
            # so use df_subset directly.
            df_valid = df_subset.dropna(axis=0, how="any")
            gmtv = pm.Data("gmt", df_valid["gmt_scaled"].values)
            xf0 = pm.Data("xf0", df_valid.filter(regex="^mode_0_").values)
            xf1 = pm.Data("xf1", df_valid.filter(regex="^mode_1_").values)

            # alpha
            b_alpha = pm.Lognormal("b_alpha", mu=1.0, sigma=1.5, testval=1)
            a_alpha = pm.Normal("a_alpha", mu=0, sigma=1.0)

            fc_alpha = pm.Normal("fc_alpha", mu=0.0, sigma=1.0, shape=xf0.dshape[1])
            fctrend_alpha = pm.Normal(
                "fctrend_alpha", mu=0.0, sigma=1.0, shape=xf1.dshape[1]
            )
            # in (-inf, inf)
            logistic = b_alpha / (
                1
                + tt.exp(
                    -1.0
                    * (
                        a_alpha * gmtv
                        + det_dot(xf0, fc_alpha)
                        + gmtv * det_dot(xf1, fctrend_alpha)
                    )
                )
            )

            alpha = pm.Deterministic("alpha", logistic)

            # beta
            b_beta = pm.Lognormal("b_beta", mu=1.0, sigma=1.5, testval=1)
            a_beta = pm.Normal("a_beta", mu=0, sigma=1.0)

            fc_beta = pm.Normal("fc_beta", mu=0.0, sigma=1.0, shape=xf0.dshape[1])
            fctrend_beta = pm.Normal(
                "fctrend_beta", mu=0.0, sigma=1.0, shape=xf1.dshape[1]
            )
            # in (-inf, inf)
            logistic = b_beta / (
                1
                + tt.exp(
                    -1.0
                    * (
                        a_beta * gmtv
                        + det_dot(xf0, fc_beta)
                        + gmtv * det_dot(xf1, fctrend_beta)
                    )
                )
            )

            beta = pm.Deterministic("beta", logistic)
            # beta = pm.HalfCauchy("beta", 0.5, testval=1)
            # beta = pm.Lognormal("beta", mu=1.0, sigma=1.5, testval=1)

            if not self.test:
                pm.Beta("obs", alpha=alpha, beta=beta, observed=df_valid["y_scaled"])

        return model


class RsdsRiceLogistic(attrici.distributions.Rice):
    """ Influence of GMT is modelled through a shift of
        mu and sigma parameters in a Beta distribution.
        """

    def __init__(self, modes):
        super(RsdsRiceLogistic, self).__init__()
        self.modes = modes
        self.test = False

    def setup(self, df_subset):
        model = pm.Model()

        with model:
            # so use df_subset directly.
            df_valid = df_subset.dropna(axis=0, how="any")
            gmtv = pm.Data("gmt", df_valid["gmt_scaled"].values)
            xf0 = pm.Data("xf0", df_valid.filter(regex="^mode_0_").values)
            xf1 = pm.Data("xf1", df_valid.filter(regex="^mode_1_").values)

            # nu
            # b_nu = pm.Lognormal("b_nu", mu=0.0, sigma=1)
            b_nu = pm.HalfCauchy("b_nu", 0.1, testval=1)
            a_nu = pm.Normal("a_nu", mu=0, sigma=0.1)

            fc_nu = pm.Normal("fc_nu", mu=0.0, sigma=1.0, shape=xf0.dshape[1])
            fctrend_nu = pm.Normal(
                "fctrend_nu", mu=0.0, sigma=0.1, shape=xf1.dshape[1]
            )
            # in (-inf, inf)
            logistic = b_nu / (
                    1
                    + tt.exp(
                -1.0
                * (
                        a_nu * gmtv
                        + det_dot(xf0, fc_nu)
                        + gmtv * det_dot(xf1, fctrend_nu)
                )
            )
            )

            nu = pm.Deterministic("nu", logistic)

            # sigma
            # b_sigma = pm.Lognormal("b_sigma", mu=0.0, sigma=1)
            b_sigma = pm.HalfCauchy("b_sigma", 0.1, testval=1)
            a_sigma = pm.Normal("a_sigma", mu=0, sigma=0.1)

            fc_sigma = pm.Normal("fc_sigma", mu=0.0, sigma=1.0, shape=xf0.dshape[1])
            # fctrend_sigma = pm.Normal(
            #     "fctrend_sigma", mu=0.0, sigma=0.1, shape=xf1.dshape[1]
            # )
            # in (-inf, inf)
            logistic = b_sigma / (
                    1
                    + tt.exp(
                -1.0
                * (
                        a_sigma * gmtv
                        + det_dot(xf0, fc_sigma)
                        # + gmtv * det_dot(xf1, fctrend_sigma)
                )
            )
            )

            sigma = pm.Deterministic("sigma", logistic)
            # sigma = pm.HalfCauchy("sigma", 0.1, testval=1)

            if not self.test:
                pm.Rice("obs", nu=nu, sigma=sigma, observed=df_valid["y_scaled"])

        return model


class RsdsRice(attrici.distributions.Rice):
    """ Influence of GMT is modelled through a shift of
        mu and sigma parameters in a Beta distribution.
        """

    def __init__(self, modes):
        super(RsdsRice, self).__init__()
        self.modes = modes
        self.test = False

    def setup(self, df_subset):
        model = pm.Model()

        with model:
            # so use df_subset directly.
            df_valid = df_subset.dropna(axis=0, how="any")
            gmtv = pm.Data("gmt", df_valid["gmt_scaled"].values)
            xf0 = pm.Data("xf0", df_valid.filter(regex="^mode_0_").values)
            xf1 = pm.Data("xf1", df_valid.filter(regex="^mode_1_").values)

            # nu
            b_nu = pm.Lognormal("b_nu", mu=-1, sigma=0.4, testval=1.0)
            a_nu = pm.Normal("a_nu", mu=0, sigma=0.05, testval=0)

            fourier_coefficients_nu = pm.Normal(
                "fourier_coefficients_nu", mu=0.0, sd=0.1, shape=xf0.dshape[1]
            )
            # in (-inf, inf)
            lin = pm.Deterministic(
                "lin_nu",
                a_nu * gmtv + b_nu + det_dot(xf0, fourier_coefficients_nu),
            )
            nu = pm.Deterministic("nu", pm.math.switch(lin>1e-3,lin, 1e-3))
            #nu = pm.Deterministic("nu", tt.nnet.elu(lin, alpha)) + 2 * alpha

            # sigma
            b_sigma = pm.Lognormal("b_sigma", mu=-1, sigma=0.4, testval=1.0)
            a_sigma = pm.Normal("a_sigma", mu=0, sigma=0.05, testval=0)

            fourier_coefficients_sigma = pm.Normal(
                "fourier_coefficients_sigma", mu=0.0, sd=0.1, shape=xf0.dshape[1]
            )
            # in (-inf, inf)
            lin = pm.Deterministic(
                "lin_sigma",
                a_sigma * gmtv + b_sigma + det_dot(xf0, fourier_coefficients_sigma),
            )
            sigma = pm.Deterministic("sigma", pm.math.switch(lin > 1e-1, lin, 1e-1))
            #sigma = pm.Deterministic("sigma", tt.nnet.elu(lin, alpha)) + 2 * alpha

            if not self.test:
                pm.Rice("obs", nu=nu, sigma=sigma, observed=df_valid["y_scaled"])

        return model


class RsdsTrendSigma(attrici.distributions.Normal):
    """ Influence of GMT is modelled through a shift of
        mu and sigma parameters in a Beta distribution.
        """

    def __init__(self, modes):
        super(RsdsTrendSigma, self).__init__()
        self.modes = modes
        self.test = False


    def quantile_mapping(self, d, y_scaled):
        """
        specific for censored normal values
        rsds can not be smaller than zero
        all values equal to zero are masked out for training
        those masked values are replaced by random values, sampled from the trained model

        those random values are quantile-mapped
        after quantile mapping negative c-fact values are mapped to zero
        """
        random_values = stats.norm.rvs(loc=d["mu"], scale=d["sigma"])
        # the random values should not be larger than 0 (any value smaller 0 is possible)
        np.minimum(random_values, 0, random_values)
        y_filled = y_scaled.fillna(pd.Series(random_values))

        quantile = stats.norm.cdf(y_filled, loc=d["mu"], scale=d["sigma"])
        x_mapped = stats.norm.ppf(quantile, loc=d["mu_ref"], scale=d["sigma_ref"])
        np.maximum(x_mapped, 0, x_mapped)
        return x_mapped

    def setup(self, df_subset):
        model = pm.Model()

        with model:
            # so use df_subset directly.
            df_valid = df_subset.dropna(axis=0, how="any")
            gmtv = pm.Data("gmt", df_valid["gmt_scaled"].values)
            xf0 = pm.Data("xf0", df_valid.filter(regex="^mode_0_").values)
            if (np.array(self.modes) > 1).any():
                raise ValueError('Modes larger 1 are not allowed for the censored model.')

            # mu
            b_mu = pm.Lognormal("b_mu", mu=-1, sigma=0.4, testval=1.0)
            a_mu = pm.Normal("a_mu", mu=0, sigma=0.05, testval=0)
            fourier_coefficients_mu = pm.Normal(
                "fourier_coefficients_mu", mu=0.0, sd=0.1, shape=xf0.dshape[1]
            )
            mu = pm.Deterministic(
                "mu",
                a_mu * gmtv + b_mu + det_dot(xf0, fourier_coefficients_mu),
            )

            # sigma
            b_sigma = pm.Lognormal("b_sigma", mu=-1, sigma=0.4, testval=1.0)
            a_sigma = pm.Normal("a_sigma", mu=0, sigma=0.05, testval=0)

            lin = pm.Deterministic(
                "lin_sigma",
                a_sigma * gmtv + b_sigma
            )
            alpha = 1e-6
            sigma = pm.Deterministic("sigma", pm.math.switch(lin > alpha, lin, alpha * tt.exp(lin-alpha)))

            if not self.test:
                pm.Normal("obs", mu=mu, sigma=sigma, observed=df_valid["y_scaled"])

        return model


class Tasrange(attrici.distributions.Normal):

    """ Influence of GMT is modelled through a shift of
    mu and sigma parameters in a Beta distribution.
    """

    def __init__(self, modes):
        super(Tasrange, self).__init__()
        self.modes = modes
        self.test = False

    def quantile_mapping(self, d, y_scaled):

        """
        specific for normally distributed variables.
        """
        quantile = stats.norm.cdf(y_scaled, loc=d["mu"], scale=d["sigma"])
        x_mapped = stats.norm.ppf(quantile, loc=d["mu_ref"], scale=d["sigma_ref"])
        x_mapped[x_mapped <= 0] = np.nan
        # values are not alowed to become non-negative. If that would happen, do not quantile map
        return x_mapped

    def setup(self, df_subset):

        model = pm.Model()

        with model:
            # so use df_subset directly.
            df_valid = df_subset.dropna(axis=0, how="any")
            gmtv = pm.Data("gmt", df_valid["gmt_scaled"].values)
            xf0 = pm.Data("xf0", df_valid.filter(regex="^mode_0_").values)
            xf1 = pm.Data("xf1", df_valid.filter(regex="^mode_1_").values)

            # mu
            # b_mu is in the interval (-inf,inf)
            b_mu = pm.Normal("b_mu", mu=0.5, sigma=1)
            # a_mu in (-inf, inf)
            a_mu = pm.Normal("a_mu", mu=0, sigma=1)

            fc_mu = pm.Normal("fc_mu", mu=0.0, sd=2.0, shape=xf0.dshape[1])
            fctrend_mu = pm.Normal("fctrend_mu", mu=0.0, sd=2.0, shape=xf1.dshape[1])

            # in (-inf, inf)
            mu = pm.Deterministic(
                "mu",
                a_mu * gmtv
                + b_mu
                + det_dot(xf0, fc_mu)
                + gmtv * det_dot(xf1, fctrend_mu),
            )

            # sigma
            sigma = pm.Lognormal("sigma", mu=-1, sigma=0.4, testval=1.0)


            if not self.test:
                pm.Normal("obs", mu=mu, sigma=sigma, observed=df_valid["y_scaled"])
        return model


class TasrangeGamma(attrici.distributions.Gamma):
    """ Influence of GMT is modelled through a shift of
    mu and sigma parameters in a Beta distribution.
    """

    def __init__(self, modes):
        super(TasrangeGamma, self).__init__()
        self.modes = modes
        self.test = False

    def setup(self, df_subset):
        model = pm.Model()

        with model:
            # so use df_subset directly.
            df_valid = df_subset.dropna(axis=0, how="any")
            gmtv = pm.Data("gmt", df_valid["gmt_scaled"].values)
            xf0 = pm.Data("xf0", df_valid.filter(regex="^mode_0_").values)

            # covariates is a [n x d] matrix with all (d-1) predictors it contains the following columns:
            # column of ones which is equivalent to the intercept term in a linear model
            # scaled gmt values
            # 2 columns (sin and cos) for each mode for the yearly cycle that does not depend on GMT
            # 2 columns (sin*GMT and cos*GMT) for each mode for the gmt-dependent change in the yearly cycle
            covariates = pm.math.concatenate(
                [
                    # tt.ones_like(gmtv)[:, None],
                    # gmtv[:, None],
                    xf0,
                    tt.tile(gmtv[:, None], (1, int(xf0.dshape[1]))) * xf0
                ],
                axis=1
            )

            # alpha_mle, beta_mle, _, _ = scipy.stats.beta.fit(df_valid["y_scaled"], floc=0, fscale=1)
            # --------------------------------------------------------
            # definition of priors

            # nu is called the precision parameter

            # nu = pm.Exponential("nu", lam=0.01)
            weights_nu_longterm_intercept = pm.Normal("weights_nu_longterm_intercept", mu=0, sd=1)
            weights_nu_fc_intercept = pm.math.concatenate(
                [pm.Normal(f"weights_nu_fc_intercept_{i}", mu=0, sd=1 / (i + 1), shape=2)
                 for i in range(int(xf0.dshape[1]) // 2)]
            )

            eta_nu = tt.dot(xf0, weights_nu_fc_intercept) + weights_nu_longterm_intercept
            nu = pm.math.exp(eta_nu)
            # weights_nu = pm.Normal("weights_nu",mu=0, sd=10, shape=2 + 2 * xf0.dshape[1])
            # eta_nu = tt.dot(covariates, weights_nu)
            # nu = pm.Deterministic("nu", pm.math.exp(eta_nu))

            # decentralized model
            # mu
            # mu_weights = pm.Normal('mu_weights', mu=0., sd=0.1)
            # sigma_weights = pm.HalfCauchy('sigma_weights', 1)

            # mu offset is a reparametrization to get a decentralized model,
            # see https://twiecki.io/blog/2017/02/08/bayesian-hierchical-non-centered/
            # and discussion in https://github.com/pymc-devs/pymc3/issues/3132
            # mu_weights_offset = pm.Normal('mu_weights_offset', mu=0, sd=10, shape=2 + 2 * xf0.dshape[1])
            # weights = pm.Deterministic("weights", mu_weights + mu_weights_offset * sigma_weights)

            # centralized model
            # weights = pm.Normal('weights', mu=mu_weights, sd=sigma_weights, shape=2+2*xf0.dshape[1])

            # nonhierarchical model (I think)
            weights_longterm_intercept = pm.Normal("weights_longterm_intercept", mu=0, sd=1)
            weights_longterm_trend = pm.Normal("weights_longterm_trend", mu=0, sd=0.1)
            weights_fc_intercept = pm.math.concatenate(
                [pm.Normal(f"weights_fc_intercept_{i}", mu=0, sd=1 / (2 * i + 1), shape=2)
                 for i in range(int(xf0.dshape[1]) // 2)]
            )
            weights_fc_trend = pm.Normal("weights_fc_trend", mu=0, sd=0.1, shape=xf0.dshape[1])
            weights_fc = pm.math.concatenate([
                weights_fc_intercept,
                weights_fc_trend
            ])
            # weights = pm.Normal("weights", mu=0, sd=1, shape=2 + 2 * xf0.dshape[1])

            # ----------------------------------------------------------
            # weights are the parameters that are learned by the model
            # eta is a linear model of the predictors
            eta = tt.dot(covariates, weights_fc) + weights_longterm_intercept + weights_longterm_trend * gmtv

            # mu models the expected value of the target data.
            # The logit function is the link function in the Generalized Linear Model
            mu = pm.Deterministic("mu", pm.math.exp(eta))
            nu = pm.math.exp(eta_nu)
            sigma = pm.Deterministic("sigma", mu * mu / nu)
            logp_ = pm.Deterministic("logp", model.logpt)
            # sigma = pm.Deterministic("sigma", mu * (1 - mu) / (1 + nu))

            # mu = tt.printing.Print('mu')(mu)
            # nu = tt.printing.Print('nu')(nu)

            if not self.test:
                pm.Gamma("obs", mu=mu, sigma=sigma, observed=df_valid["y_scaled"])

        return model


class TasrangeSigmaTrend(attrici.distributions.Normal):

    """ Influence of GMT is modelled through a shift of
    mu and sigma parameters in a Beta distribution.
    """

    def __init__(self, modes):
        super(TasrangeSigmaTrend, self).__init__()
        self.modes = modes
        self.test = False

    def quantile_mapping(self, d, y_scaled):

        """
        specific for normally distributed variables.
        """
        quantile = stats.norm.cdf(y_scaled, loc=d["mu"], scale=d["sigma"])
        x_mapped = stats.norm.ppf(quantile, loc=d["mu_ref"], scale=d["sigma_ref"])
        x_mapped[x_mapped <= 0] = np.nan
        # values are not alowed to become non-negative. If that would happen, do not quantile map
        return x_mapped

    def setup(self, df_subset):

        model = pm.Model()

        with model:
            # so use df_subset directly.
            df_valid = df_subset.dropna(axis=0, how="any")
            gmtv = pm.Data("gmt", df_valid["gmt_scaled"].values)
            xf0 = pm.Data("xf0", df_valid.filter(regex="^mode_0_").values)
            xf1 = pm.Data("xf1", df_valid.filter(regex="^mode_1_").values)

            # mu
            # b_mu is in the interval (-inf,inf)
            b_mu = pm.Normal("b_mu", mu=0.5, sigma=1)
            # a_mu in (-inf, inf)
            a_mu = pm.Normal("a_mu", mu=0, sigma=1)

            fc_mu = pm.Normal("fc_mu", mu=0.0, sd=2.0, shape=xf0.dshape[1])
            fctrend_mu = pm.Normal("fctrend_mu", mu=0.0, sd=2.0, shape=xf1.dshape[1])

            # in (-inf, inf)
            mu = pm.Deterministic(
                "mu",
                a_mu * gmtv
                + b_mu
                + det_dot(xf0, fc_mu)
                + gmtv * det_dot(xf1, fctrend_mu),
            )

            # sigma
            # b_sigma = pm.Lognormal("b_sigma", mu=-1, sigma=0.4, testval=1.0)
            # a_sigma = pm.Normal("a_sigma", mu=0, sigma=0.05, testval=0)
            #
            # lin = pm.Deterministic(
            #     "lin_sigma",
            #     a_sigma * gmtv + b_sigma
            # )
            b_sigma = pm.Normal("b_sigma", mu=1
                                , sigma=0.4, testval=1.0)
            a_sigma = pm.Normal("a_sigma", mu=0, sigma=0.05, testval=0)

            lin = pm.Deterministic(
                "lin_sigma",
                a_sigma * mu + b_sigma
            )
            alpha = 1e-6
            sigma = pm.Deterministic("sigma", pm.math.switch(lin > alpha, lin, alpha))

            if not self.test:
                pm.Normal("obs", mu=mu, sigma=sigma, observed=df_valid["y_scaled"])
        return model


class TasrangeRice(attrici.distributions.Rice):

    """ Influence of GMT is modelled through a shift of
    mu and sigma parameters in a Beta distribution.
    """

    def __init__(self, modes):
        super(TasrangeRice, self).__init__()
        self.modes = modes
        self.test = False

    def setup(self, df_subset):

        model = pm.Model()

        with model:
            # so use df_subset directly.
            df_valid = df_subset.dropna(axis=0, how="any")
            gmtv = pm.Data("gmt", df_valid["gmt_scaled"].values)
            xf0 = pm.Data("xf0", df_valid.filter(regex="^mode_0_").values)
            xf1 = pm.Data("xf1", df_valid.filter(regex="^mode_1_").values)

            # nu
            # b_nu = pm.HalfCauchy("b_nu", 0.1,testval=0.5)
            b_nu = pm.Lognormal("b_nu", mu=0, sigma=.5, testval=1)
            a_nu = pm.Normal("a_nu", mu=0, sigma=0.05, testval=0)

            lin = pm.Deterministic(
                "lin_nu",
                a_nu * gmtv + b_nu
            )

            nu = pm.Deterministic("nu", pm.math.switch(lin > 1e-6, lin, 1e-6))
            # nu = pm.HalfCauchy("nu", .1, testval=.5)

            # sigma
            # sigma = pm.Lognormal("sigma", mu=1, sigma=.5, testval=1)
            b_sigma = pm.Lognormal("b_sigma", mu=0, sigma=.5, testval=1)
            a_sigma = pm.Normal("a_sigma", mu=0, sigma=0.01, testval=0)

            fc_sigma = pm.Normal("fc_sigma", mu=0.0, sigma=.1, shape=xf0.dshape[1], testval=0)
            fctrend_sigma = pm.Normal("fctrend_sigma", mu=0.0, sigma=0.1, shape=xf1.dshape[1],testval=0)

            lin_sigma = pm.Deterministic(
                "lin_sigma",
                (a_sigma + det_dot(xf1, fctrend_sigma)) * gmtv + b_sigma + det_dot(xf0, fc_sigma),
            )
            cutoff = tt.sqrt(nu)/5 # otherwise x*nu/sigma**2 gets larger then 25 which may lead to inf value in logp
            #cutoff = 1e-3
            sigma = pm.Deterministic("sigma", pm.math.switch(lin_sigma > cutoff, lin_sigma, cutoff))

            if not self.test:
                pm.Rice("obs", nu=nu, sigma=sigma, observed=df_valid["y_scaled"])

        return model


class TasrangeRiceConstSigma(attrici.distributions.Rice):

    """ Influence of GMT is modelled through a shift of
    mu and sigma parameters in a Beta distribution.
    """

    def __init__(self, modes):
        super(TasrangeRiceConstSigma, self).__init__()
        self.modes = modes
        self.test = False

    def setup(self, df_subset):

        model = pm.Model()

        with model:
            # so use df_subset directly.
            df_valid = df_subset.dropna(axis=0, how="any")
            gmtv = pm.Data("gmt", df_valid["gmt_scaled"].values)
            xf0 = pm.Data("xf0", df_valid.filter(regex="^mode_0_").values)
            xf1 = pm.Data("xf1", df_valid.filter(regex="^mode_1_").values)

            # nu
            # b_nu = pm.Lognormal("b_nu", mu=0.0, sigma=1)
            b_nu = pm.HalfCauchy("b_nu", 0.1, testval=1)
            a_nu = pm.Normal("a_nu", mu=0, sigma=0.1)

            fc_nu = pm.Normal("fc_nu", mu=0.0, sigma=1.0, shape=xf0.dshape[1])
            fctrend_nu = pm.Normal("fctrend_nu", mu=0.0, sigma=0.1, shape=xf1.dshape[1])

            lin_nu = pm.Deterministic(
                "lin_nu",
                (a_nu + det_dot(xf1, fctrend_nu)) * gmtv + b_nu + det_dot(xf0, fc_nu),
            )
            cutoff = 1e-6
            nu = pm.Deterministic("nu", pm.math.switch(lin_nu > cutoff, lin_nu, cutoff))
            # sigma
            sigma = pm.HalfCauchy("sigma", 0.1, testval=1)

            if not self.test:
                pm.Rice("obs", nu=nu, sigma=sigma, observed=df_valid["y_scaled"])

        return model


class TasrangeLogistic(attrici.distributions.Rice):

    """ Influence of GMT is modelled through a shift of
    mu and sigma parameters in a Beta distribution.
    """

    def __init__(self, modes):
        super(TasrangeLogistic, self).__init__()
        self.modes = modes
        self.test = False

    def setup(self, df_subset):

        model = pm.Model()

        with model:
            # so use df_subset directly.
            df_valid = df_subset.dropna(axis=0, how="any")
            gmtv = pm.Data("gmt", df_valid["gmt_scaled"].values)
            xf0 = pm.Data("xf0", df_valid.filter(regex="^mode_0_").values)
            xf1 = pm.Data("xf1", df_valid.filter(regex="^mode_1_").values)

            # nu
            # b_nu = pm.Lognormal("b_nu", mu=0.0, sigma=1)
            b_nu = pm.HalfCauchy("b_nu", 0.1, testval=1)
            a_nu = pm.Normal("a_nu", mu=0, sigma=0.1)

            fc_nu = pm.Normal("fc_nu", mu=0.0, sigma=1.0, shape=xf0.dshape[1])
            fctrend_nu = pm.Normal("fctrend_nu", mu=0.0, sigma=0.1, shape=xf1.dshape[1])
            # in (-inf, inf)
            logistic = b_nu / (
                1
                + tt.exp(
                    -1.0
                    * (
                        a_nu * gmtv
                        + det_dot(xf0, fc_nu)
                        + gmtv * det_dot(xf1, fctrend_nu)
                    )
                )
            )

            nu = pm.Deterministic("nu", logistic)
            sigma = pm.HalfCauchy("sigma", 0.1, testval=1)

            if not self.test:
                pm.Rice("obs", nu=nu, sigma=sigma, observed=df_valid["y_scaled"])

        return model


class Wind(attrici.distributions.Weibull):

    """ Influence of GMT is modelled through a shift of
    the scale parameter beta in the Weibull distribution. The shape
    parameter alpha is assumed free of a trend.

    """

    def __init__(self, modes):
        super(Wind, self).__init__()
        self.modes = modes
        self.test = False

    def setup(self, df_subset):

        model = pm.Model()

        with model:
            # so use df_subset directly.
            df_valid = df_subset.dropna(axis=0, how="any")
            gmtv = pm.Data("gmt", df_valid["gmt_scaled"].values)
            xf0 = pm.Data("xf0", df_valid.filter(regex="^mode_0_").values)
            xf1 = pm.Data("xf1", df_valid.filter(regex="^mode_1_").values)

            # beta
            b_beta = pm.Lognormal("b_beta", mu=-1, sigma=0.4, testval=1.0)
            a_beta = pm.Normal("a_beta", mu=0, sigma=.01)

            fc_beta = pm.Normal("fc_beta", mu=0.0, sd=.01, shape=xf0.dshape[1])
            fctrend_beta = pm.Normal("fctrend_beta", mu=0.0, sd=.01, shape=xf1.dshape[1])

            lin_beta = pm.Deterministic(
                "lin_beta",
                (a_beta + det_dot(xf1, fctrend_beta)) * gmtv
                + b_beta
                + det_dot(xf0, fc_beta)
            )
            cutoff = 1e-10
            beta = pm.Deterministic("beta", pm.math.switch(lin_beta > cutoff, lin_beta, cutoff))

            # alpha
            alpha = pm.Lognormal("alpha", mu=-1, sigma=0.4, testval=1.0)

            if not self.test:
                pm.Weibull("obs", alpha=alpha, beta=beta, observed=df_valid["y_scaled"])

        return model


class WindFull(attrici.distributions.Weibull):

    """ Influence of GMT is modelled through a shift of
    the scale parameter beta in the Weibull distribution. The shape
    parameter alpha is assumed to be a function of beta.

    """

    def __init__(self, modes):
        super(WindFull, self).__init__()
        self.modes = modes
        self.test = False

    def setup(self, df_subset):

        model = pm.Model()

        with model:
            # so use df_subset directly.
            df_valid = df_subset.dropna(axis=0, how="any")
            gmtv = pm.Data("gmt", df_valid["gmt_scaled"].values)
            xf0 = pm.Data("xf0", df_valid.filter(regex="^mode_0_").values)
            xf1 = pm.Data("xf1", df_valid.filter(regex="^mode_1_").values)

            # beta
            b_beta = pm.Lognormal("b_beta", mu=-1, sigma=0.4, testval=1.0)
            a_beta = pm.Normal("a_beta", mu=0, sigma=.01)

            fc_beta = pm.Normal("fc_beta", mu=0.0, sd=.01, shape=xf0.dshape[1])
            fctrend_beta = pm.Normal("fctrend_beta", mu=0.0, sd=.01, shape=xf1.dshape[1])

            lin_beta = pm.Deterministic(
                "lin_beta",
                (a_beta + det_dot(xf1, fctrend_beta)) * gmtv
                + b_beta
                + det_dot(xf0, fc_beta)
            )
            cutoff = 1e-10
            beta = pm.Deterministic("beta", pm.math.switch(lin_beta > cutoff, lin_beta, cutoff))

            # alpha
            # a alpha and b alpha could also be a function of GMT
            a_alpha = pm.Normal("a_alpha", mu=0.0, sd=.1)
            b_alpha = pm.Normal("b_alpha", mu=2, sd=1)
            lin_alpha = a_alpha * beta + b_alpha
            cutoff = 1e-10
            alpha = pm.Deterministic("alpha", pm.math.switch(lin_alpha > cutoff, lin_alpha, cutoff))

            if not self.test:
                pm.Weibull("obs", alpha=alpha, beta=beta, observed=df_valid["y_scaled"])

        return model


class WindLogistic(attrici.distributions.Weibull):

    """ Influence of GMT is modelled through a shift of
    the scale parameter beta in the Weibull distribution. The shape
    parameter alpha is assumed free of a trend.

    """

    def __init__(self, modes):
        super(WindLogistic, self).__init__()
        self.modes = modes
        self.test = False

    def setup(self, df_subset):

        model = pm.Model()

        with model:
            # so use df_subset directly.
            df_valid = df_subset.dropna(axis=0, how="any")
            gmtv = pm.Data("gmt", df_valid["gmt_scaled"].values)
            xf0 = pm.Data("xf0", df_valid.filter(regex="^mode_0_").values)
            xf1 = pm.Data("xf1", df_valid.filter(regex="^mode_1_").values)

            b_beta = pm.HalfCauchy("b_beta", 0.1, testval=1)
            a_beta = pm.Normal("a_beta", mu=0, sigma=0.1)

            fc_beta = pm.Normal("fc_beta", mu=0.0, sigma=1.0, shape=xf0.dshape[1])
            fctrend_beta = pm.Normal(
                "fctrend_beta", mu=0.0, sigma=0.1, shape=xf1.dshape[1]
            )
            # in (-inf, inf)
            logistic = b_beta / (
                1
                + tt.exp(
                    -1.0
                    * (
                        a_beta * gmtv
                        + det_dot(xf0, fc_beta)
                        + gmtv * det_dot(xf1, fctrend_beta)
                    )
                )
            )

            beta = pm.Deterministic("beta", logistic)
            alpha = pm.HalfCauchy("alpha", 0.1, testval=1)

            if not self.test:
                pm.Weibull("obs", alpha=alpha, beta=beta, observed=df_valid["y_scaled"])

        return model