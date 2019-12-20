import numpy as np
import pymc3 as pm
from scipy import stats
import pandas as pd
import theano.tensor as tt
import icounter.logistic as l


class Normal(object):

    """ Influence of GMT is modelled through a shift of
    mu and sigma parameters in a Normal distribution.
    Works for example for tas. """

    def __init__(self, modes, mu_model, sigma_model):

        self.modes = modes
        self.mu_model = mu_model
        self.sigma_model = sigma_model

        print("Using Normal distribution model. Fourier modes:", modes)

    def setup(self, df_valid):

        model = pm.Model()

        with model:

            gmt = pm.Data("gmt", df_valid["gmt_scaled"].values)
            xf0 = pm.Data("xf0", df_valid.filter(like="mode_0_").values)
            xf1 = pm.Data("xf1", df_valid.filter(like="mode_1_").values)
            xf2 = pm.Data("xf2", df_valid.filter(like="mode_2_").values)

            mu = l.full(model, pm.Normal, "mu", gmt, xf0, xf1, ic_sigma=5.0)

            if self.sigma_model == "full":
                xf3 = pm.Data("xf3", df_valid.filter(like="mode_3_").values)
                sigma = l.full(model, pm.Lognormal, "sigma", gmt, xf2, xf3)

            elif self.sigma_model == "yearlycycle":
                sigma = l.yearlycycle(model, pm.Lognormal, "sigma", xf2)

            elif self.sigma_model == "longterm_yearlycycle":
                sigma = l.longterm_yearlycycle(model, pm.Lognormal, "sigma", gmt, xf2)

            elif self.sigma_model == "longterm":
                sigma = l.longterm(model, pm.Lognormal, "sigma", gmt)

            else:
                raise NotImplemented

            pm.Normal("obs", mu=mu, sigma=sigma, observed=df_valid["y_scaled"])

        return model

    def quantile_mapping(self, d, y_scaled):

        """
        specific for normally distributed variables.
        """
        quantile = stats.norm.cdf(y_scaled, loc=d["mu"], scale=d["sigma"])
        x_mapped = stats.norm.ppf(quantile, loc=d["mu_ref"], scale=d["sigma_ref"])

        return x_mapped


class GammaBernoulli(object):

    """ Influence of GMT is modelled through the parameters of the Gamma
    distribution. Example: precipitation """

    def __init__(self, modes, mu_model, sigma_model, bernoulli_model):

        self.modes = modes
        self.mu_model = mu_model
        self.sigma_model = sigma_model
        self.bernoulli_model = bernoulli_model

        print("Using GammaBernoulli distribution model. Fourier modes:", modes)

    def setup(self, df_valid, df):

        model = pm.Model()

        with model:

            gmt = pm.Data("gmt", df["gmt_scaled"].values)
            xf0 = pm.Data("xf0", df.filter(like="mode_0_").values)
            xf1 = pm.Data("xf1", df.filter(like="mode_1_").values)
            xf2 = pm.Data("xf2", df.filter(like="mode_2_").values)
            xf3 = pm.Data("xf3", df.filter(like="mode_3_").values)

            # FIXME: avoid this doubling of predictors, introduced for BernoulliGamma
            gmtv = pm.Data("gmtv", df_valid["gmt_scaled"].values)
            xf0v = pm.Data("xf0v", df_valid.filter(like="mode_0_").values)
            xf1v = pm.Data("xf1v", df_valid.filter(like="mode_1_").values)
            xf2v = pm.Data("xf2v", df_valid.filter(like="mode_2_").values)
            xf3v = pm.Data("xf3v", df_valid.filter(like="mode_3_").values)

            if self.bernoulli_model == "full":
                raise ValueError(f'bernoulli model {self.bernoulli_model} is not implemented yet')
                pbern = l.full(
                    model, pm.Beta, "pbern", gmt, xf0, xf1, ic_mu=0.5, ic_sigma=0.1, ic_fac=2.0
                )

            elif self.bernoulli_model == "yearlycycle":
                raise ValueError(f'bernoulli model {self.bernoulli_model} is not implemented yet')
                pbern = l.yearlycycle(
                    model, pm.Beta, "pbern", xf0, ic_mu=0.5, ic_sigma=0.1, ic_fac=2.0
                )

            elif self.bernoulli_model == "longterm_yearlycycle":
                raise ValueError(f'bernoulli model {self.bernoulli_model} is not implemented yet')
                pbern = l.longterm_yearlycycle(
                    model, pm.Beta("pbern_intercept",alpha=2,beta=0.5), "pbern", gmt, xf0,
                )

            elif self.bernoulli_model == "longterm":
                # todo outsource params
                # b is in the interval (0,2)
                b = pm.Beta("pbern_b", alpha=2, beta=2)  # beta(2, 2) is symmetric with mode at 0.5 b is in
                # a is in the interval (-b,1-b)
                a = tt.sub(pm.Beta("pbern_a", alpha=2, beta=2), b)
                # pbern is in the interval (0,1)
                pbern = a * gmt + b                 # pbern is a linear model of gmt
                pbern = pm.Deterministic('pbern', pbern) # todo: unit test to test whether pbern in (0,1) for any alpha,beta

            else:
                raise NotImplemented

            if self.mu_model == "full":
                raise ValueError(f'bernoulli model {self.bernoulli_model} is not implemented yet')
                mu = l.full(model, pm.Lognormal, "mu", gmtv, xf0v, xf1v)

            elif self.mu_model == "yearlycycle":
                raise ValueError(f'bernoulli model {self.bernoulli_model} is not implemented yet')
                mu = l.yearlycycle(model, pm.Lognormal, "mu", xf0v)

            elif self.mu_model == "longterm_yearlycycle":
                raise ValueError(f'bernoulli model {self.bernoulli_model} is not implemented yet')
                mu = l.longterm_yearlycycle(model, pm.Lognormal("mu_intercept",mu=0,sigma=1), "mu", gmtv, xf0v)

            elif self.mu_model == "longterm":
                with model:
                    # b is in the interval (0,inf)
                    b = pm.Exponential('mu_b', lam=1)
                    # a is in the interval (-b, inf)
                    a = tt.sub(pm.Exponential('mu_a', lam=1), b)   # a in (-b, inf)
                    mu = a * gmtv + b        # in (0, inf)
                mu = pm.Deterministic('mu', mu)

            else:
                raise NotImplemented

            if self.sigma_model == "full":
                raise ValueError(f'bernoulli model {self.bernoulli_model} is not implemented yet')
                sigma = l.full(model, pm.Lognormal, "sigma", gmtv, xf2v, xf3v)

            elif self.sigma_model == "yearlycycle":
                raise ValueError(f'bernoulli model {self.bernoulli_model} is not implemented yet')
                sigma = l.yearlycycle(model, pm.Lognormal, "sigma", xf2v)

            elif self.sigma_model == "longterm_yearlycycle":
                raise ValueError(f'bernoulli model {self.bernoulli_model} is not implemented yet')
                sigma = l.longterm_yearlycycle(model, pm.Lognormal("sigma_intercept",mu=0,sigma=1), "sigma", gmtv, xf2v)

            elif self.sigma_model == "longterm":
                with model:
                    # b is in the interval (0,inf)
                    mu = 0
                    sigma = 0.25
                    b = pm.Lognormal('sigma_b', mu=mu, sigma=sigma)  # mode close to one
                    # a is in the interval (-b, inf)
                    a = tt.sub(pm.Lognormal('sigma_a', mu=mu, sigma=sigma), b)   # a in (-b, inf), mode at 0
                    sigma = a * gmtv + b        # in (0, inf)
                sigma = pm.Deterministic('sigma', sigma)

            else:
                raise NotImplemented

            pm.Bernoulli("bernoulli", p=pbern, observed=df["is_dry_day"].astype(int))
            pm.Gamma("obs", mu=mu, sigma=sigma, observed=df_valid["y_scaled"])

        return model

    def quantile_mapping(self, d, y_scaled):

        """ Needs a thorough description of QM for BernoulliGamma """

        def bgamma_cdf(d, y_scaled):

            quantile = d["pbern"] + (1 - d["pbern"]) * stats.gamma.cdf(
                y_scaled,
                d["mu"] ** 2.0 / d["sigma"] ** 2.0,
                scale=d["sigma"] ** 2.0 / d["mu"],
            )
            return quantile

        def bgamma_ppf(d, quantile):

            x_mapped = stats.gamma.ppf(
                (quantile - d["pbern_ref"]) / (1 - d["pbern_ref"]),
                d["mu_ref"] ** 2.0 / d["sigma_ref"] ** 2.0,
                scale=d["sigma_ref"] ** 2.0 / d["mu_ref"],
            )

            return x_mapped

        # make it a numpy array, so we can compine smoothly with d data frame.
        y_scaled = y_scaled.values.copy()
        dry_day = np.isnan(y_scaled)
        # FIXME: have this zero precip at dry day fix earlier (in const.py for example)
        y_scaled[dry_day] = 0
        quantile = bgamma_cdf(d, y_scaled)

        # case of p smaller p'
        # the probability of a dry day is higher in the counterfactual day
        # than in the historical day. We need to create dry days.
        drier_cf = d["pbern_ref"] > d["pbern"]
        wet_to_wet = quantile > d["pbern_ref"]  # False on dry day (NA in y_scaled)
        # if the quantile of the observed rain is high enough, keep day wet
        # and use normal quantile mapping
        do_normal_qm_0 = np.logical_and(drier_cf, wet_to_wet)
        print("normal qm for higher cfact dry probability:", do_normal_qm_0.sum())
        cfact = np.zeros(len(y_scaled))
        cfact[do_normal_qm_0] = bgamma_ppf(d, quantile)[do_normal_qm_0]
        # else: make it a dry day with zero precip (from np.zeros)

        # case of p' smaller p
        # the probability of a dry day is lower in the counterfactual day
        # than in the historical day. We need to create wet days.
        wetter_cf = ~drier_cf
        wet_day = ~dry_day
        # wet days stay wet, and are normally quantile mapped
        do_normal_qm_1 = np.logical_and(wetter_cf, wet_day)
        print("normal qm for higher cfact wet probability:", do_normal_qm_1.sum())
        cfact[do_normal_qm_1] = bgamma_ppf(d, quantile)[do_normal_qm_1]
        # some dry days need to be made wet. take a random quantile from
        # the quantile range that was dry days before
        random_dry_day_q = np.random.rand(len(d)) * d["pbern"]
        map_to_wet = random_dry_day_q > d["pbern_ref"]
        # map these dry days to wet, which are not dry in obs and
        # wet in counterfactual
        randomly_map_to_wet = np.logical_and(~do_normal_qm_1, map_to_wet)
        cfact[randomly_map_to_wet] = bgamma_ppf(d, quantile)[randomly_map_to_wet]
        # else: leave zero (from np.zeros)
        print("Days originally dry:", dry_day.sum())
        print("Days made wet:", randomly_map_to_wet.sum())
        print("Days in cfact dry:", (cfact == 0).sum())
        print(
            "Total days:",
            (cfact == 0).sum()
            + randomly_map_to_wet.sum()
            + do_normal_qm_0.sum()
            + do_normal_qm_1.sum(),
        )

        return cfact


class Beta(object):

    """ Influence of GMT is modelled through the influence of on the alpha parameter
    of a Beta distribution. Beta parameter is assumed free of a trend. """

    def __init__(self, modes, sigma_model):

        self.modes = modes
        self.sigma_model = sigma_model
        self.vars_to_estimate = [
            "alpha_intercept",
            "alpha_slope",
            "alpha_yearly",
            "alpha_trend",
            "beta_intercept",
            "beta_slope",
            "beta_yearly",
            "beta_trend",
        ]

        print("Using Beta distribution model.")

    def setup(self, gmt_valid, x_fourier, observed):

        model = pm.Model()

        with model:

            gmt = pm.Data("gmt", gmt_valid)
            xf0 = pm.Data("xf0", x_fourier[0])
            xf1 = pm.Data("xf1", x_fourier[1])
            xf2 = pm.Data("xf2", x_fourier[2])
            xf3 = pm.Data("xf3", x_fourier[3])
            alpha_intercept = pm.Lognormal("alpha_intercept", mu=4, sigma=1.6)
            alpha_slope = pm.Normal("alpha_slope", mu=0, sigma=2.0)
            alpha_yearly = pm.Normal(
                "alpha_yearly", mu=0.0, sd=5.0, shape=2 * self.modes[0]
            )
            alpha_trend = pm.Normal(
                "alpha_trend", mu=0.0, sd=2.0, shape=2 * self.modes[1]
            )
            # alpha_intercept * logistic(gmt,yearly_cycle), strictly positive
            alpha = pm.Deterministic(
                "alpha",
                alpha_intercept
                / (
                    1
                    + tt.exp(
                        -1
                        * (
                            alpha_slope * gmt
                            + det_dot(xf0, alpha_yearly)
                            + gmt * det_dot(xf1, alpha_trend)
                        )
                    )
                ),
            )

            beta_intercept = pm.Lognormal("beta_intercept", mu=4, sigma=1.6)
            beta_slope = pm.Normal("beta_slope", mu=0, sigma=2.0)
            beta_yearly = pm.Normal(
                "beta_yearly", mu=0.0, sd=5.0, shape=2 * self.modes[2]
            )
            beta_trend = pm.Normal(
                "beta_trend", mu=0.0, sd=2.0, shape=2 * self.modes[3]
            )
            # beta_intercept * logistic(gmt,yearly_cycle), strictly positive
            beta = pm.Deterministic(
                "beta",
                beta_intercept
                / (
                    1
                    + tt.exp(
                        -1
                        * (
                            beta_slope * gmt
                            + det_dot(xf2, beta_yearly)
                            + gmt * det_dot(xf3, beta_trend)
                        )
                    )
                ),
            )

            # sg_intercept = pm.Lognormal("sg_intercept", mu=0, sigma=1.0)
            # sg_yearly = pm.Normal(
            #         "sg_yearly", mu=0.0, sd=5.0, shape=2 * self.modes[2]
            #     )
            # sg_intercept * logistic(gmt,yearly_cycle), strictly positive
            # sigma = pm.Beta("sigma", mu=0.5, sigma=0.2)
            # mu = pm.Beta("mu", mu=0.5, sigma=0.2)

            # kappa = mu * (1 - mu) / sigma**2. - 1.

            # alpha = pm.Deterministic("alpha",mu*kappa)
            # beta = pm.Deterministic("beta",(1.-mu)*kappa)
            mu = pm.Deterministic("mu", alpha / (alpha + beta))
            sigma = pm.Deterministic(
                "sigma",
                (alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))) ** 0.5,
            )

            pm.Beta("obs", alpha=alpha, beta=beta, observed=observed)

        return model

    def pm_sigma1(self, sg_intercept, xf2, sg_yearly):

        return sg_intercept / (1 + tt.exp(-1 * det_dot(xf2, sg_yearly)))

    def quantile_mapping(self, d, y_scaled):

        """
        specific for normally distributed variables. Mapping done for each day.
        """
        alpha = d["mu"] ** 2 * ((1 - d["mu"]) / d["sigma"] ** 2 - 1 / d["mu"])
        alpha_ref = d["mu_ref"] ** 2 * (
            (1 - d["mu_ref"]) / d["sigma_ref"] ** 2 - 1 / d["mu_ref"]
        )

        beta = alpha * (1 / d["mu"] - 1)
        beta_ref = alpha_ref * (1 / d["mu_ref"] - 1)

        quantile = stats.beta.cdf(y_scaled, alpha, beta)
        x_mapped = stats.beta.ppf(quantile, alpha_ref, beta_ref)

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

    def __init__(self, modes, sigma_model):

        self.modes = modes
        self.sigma_model = sigma_model
        self.vars_to_estimate = [
            "alpha_intercept",
            "alpha_slope",
            "alpha_yearly",
            "alpha_trend",
            "beta_intercept",
            "beta_slope",
            "beta_yearly",
            "beta_trend",
        ]

        print("Using Rice distribution model.")

    def setup(self, gmt_valid, x_fourier, observed):

        model = pm.Model()

        with model:

            gmt = pm.Data("gmt", gmt_valid)
            xf0 = pm.Data("xf0", x_fourier[0])
            xf1 = pm.Data("xf1", x_fourier[1])
            xf2 = pm.Data("xf2", x_fourier[2])
            xf3 = pm.Data("xf3", x_fourier[3])
            nu_intercept = pm.Lognormal("nu_intercept", mu=4, sigma=1.6)
            nu_slope = pm.Normal("nu_slope", mu=0, sigma=2.0)
            nu_yearly = pm.Normal("nu_yearly", mu=0.0, sd=5.0, shape=2 * self.modes[0])
            nu_trend = pm.Normal("nu_trend", mu=0.0, sd=2.0, shape=2 * self.modes[1])
            # alpha_intercept * logistic(gmt,yearly_cycle), strictly positive
            nu = pm.Deterministic(
                "mu",
                nu_intercept
                / (
                    1
                    + tt.exp(
                        -1
                        * (
                            nu_slope * gmt
                            + det_dot(xf0, nu_yearly)
                            + gmt * det_dot(xf1, nu_trend)
                        )
                    )
                ),
            )

            sigma_intercept = pm.Lognormal("sigma_intercept", mu=4, sigma=1.6)
            sigma_slope = pm.Normal("sigma_slope", mu=0, sigma=2.0)
            sigma_yearly = pm.Normal(
                "sigma_yearly", mu=0.0, sd=5.0, shape=2 * self.modes[2]
            )
            sigma_trend = pm.Normal(
                "sigma_trend", mu=0.0, sd=2.0, shape=2 * self.modes[3]
            )
            # sigma_intercept * logistic(gmt,yearly_cycle), strictly positive
            sigma = pm.Deterministic(
                "sigma",
                sigma_intercept
                / (
                    1
                    + tt.exp(
                        -1
                        * (
                            sigma_slope * gmt
                            + det_dot(xf2, sigma_yearly)
                            + gmt * det_dot(xf3, sigma_trend)
                        )
                    )
                ),
            )

            pm.Rice("obs", nu=nu, sigma=sigma, observed=observed)

        return model

    def quantile_mapping(self, d, y_scaled):

        """
        specific for normally distributed variables. Mapping done for each day.
        """
        quantile = stats.rice.cdf(y_scaled, b=d["mu"] / d["sigma"], scale=d["sigma"])
        x_mapped = stats.rice.ppf(
            quantile, b=d["mu_ref"] / d["sigma_ref"], scale=d["sigma_ref"]
        )

        return x_mapped
