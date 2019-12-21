import numpy as np
import pymc3 as pm
from scipy import stats
import pandas as pd
import theano.tensor as tt


class GammaBernoulli(object):

    """ Influence of GMT is modelled through the parameters of the Gamma
    distribution. Example: precipitation """

    def __init__(self, modes, mu_model, sigma_model, bernoulli_model):

        self.modes = modes
        self.mu_model = mu_model
        self.sigma_model = sigma_model
        self.bernoulli_model = bernoulli_model
        self.test = False

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
                raise ValueError(
                    f"bernoulli model {self.bernoulli_model} is not implemented yet"
                )
                pbern = l.full(
                    model,
                    pm.Beta,
                    "pbern",
                    gmt,
                    xf0,
                    xf1,
                    ic_mu=0.5,
                    ic_sigma=0.1,
                    ic_fac=2.0,
                )

            elif self.bernoulli_model == "yearlycycle":
                raise ValueError(
                    f"bernoulli model {self.bernoulli_model} is not implemented yet"
                )
                pbern = l.yearlycycle(
                    model, pm.Beta, "pbern", xf0, ic_mu=0.5, ic_sigma=0.1, ic_fac=2.0
                )

            elif self.bernoulli_model == "longterm_yearlycycle":
                # FIXME: very slow so far.
                # Unclear if we need scaling of c_yearly
                # Does not provide improved results as compared to longterm,
                # though more than 25x the CPU time
                pbern_fourier_coeffs = pm.Beta(
                    "pbern_fourier_coeffs", alpha=2, beta=3, shape=xf0.dshape[1]
                )
                b_const = pm.Beta("pbern_b", alpha=2, beta=2)
                c_yearly = l.det_dot(xf0 / 2.0 + 0.5, pbern_fourier_coeffs)
                b_scale = pm.Beta("b_scale", alpha=0.5, beta=1.0) * (
                    1 - b_const
                )  # /tt.max(c_yearly)
                b = pm.Deterministic("b", b_const + b_scale * c_yearly)
                # pp = pm.Deterministic("ttmax",tt.max(c_yearly))
                # a is in the interval (-b,1-b)
                a = pm.Deterministic("a", pm.Beta("pbern_a", alpha=2, beta=2) - b)
                pbern = pm.Deterministic("pbern", a * gmt + b)

            elif self.bernoulli_model == "longterm":
                # b is in the interval (0,2)
                b = pm.Beta(
                    "pbern_b", alpha=2, beta=2
                )  # beta(2, 2) is symmetric with mode at 0.5 b is in
                # a is in the interval (-b,1-b)
                a = tt.sub(pm.Beta("pbern_a", alpha=2, beta=2), b)
                # pbern is in the interval (0,1)
                pbern = a * gmt + b  # pbern is a linear model of gmt
                pbern = pm.Deterministic(
                    "pbern", pbern
                )  # todo: unit test to test whether pbern in (0,1) for any alpha,beta

            else:
                raise NotImplemented

            if self.mu_model == "full":
                raise ValueError(
                    f"bernoulli model {self.bernoulli_model} is not implemented yet"
                )
                mu = l.full(model, pm.Lognormal, "mu", gmtv, xf0v, xf1v)

            elif self.mu_model == "yearlycycle":
                raise ValueError(
                    f"bernoulli model {self.bernoulli_model} is not implemented yet"
                )
                mu = l.yearlycycle(model, pm.Lognormal, "mu", xf0v)

            elif self.mu_model == "longterm_yearlycycle":
                # raise ValueError(f'bernoulli model {self.bernoulli_model} is not implemented yet')

                mu_fourier_coeffs = pm.Exponential(
                    "mu_fourier_coeffs", lam=1, shape=xf0v.dshape[1]
                )
                # b_const = pm.Beta("pbern_b", alpha=2, beta=2)
                b_yearly = l.det_dot(xf0v / 2.0 + 0.5, mu_fourier_coeffs)
                b_const = pm.Exponential("b_const_mu", lam=1)
                b_mu = pm.Deterministic("b_mu", b_const + b_yearly)
                a_mu = tt.sub(pm.Exponential("a_mu", lam=1), b_mu)  # a in (-b, inf)
                # mu = a * gmtv + b  # in (0, inf)
                mu = pm.Deterministic("mu", a_mu * gmtv + b_mu)
                # mu = l.longterm_yearlycycle(
                #     model, pm.Lognormal("mu_intercept", mu=0, sigma=1), "mu", gmtv, xf0v
                # )

            elif self.mu_model == "longterm":
                # b_mu is in the interval (0,inf)
                b_mu = pm.Exponential("b_mu", lam=1)
                # a_mu in (-b, inf)
                a_mu = pm.Deterministic(
                    "a_mu", pm.Exponential("am", lam=1) - b_mu
                )
                mu = pm.Deterministic("mu", a_mu * gmtv + b_mu)  # in (0, inf)

            else:
                raise NotImplemented

            if self.sigma_model == "full":
                raise ValueError(
                    f"bernoulli model {self.bernoulli_model} is not implemented yet"
                )
                sigma = l.full(model, pm.Lognormal, "sigma", gmtv, xf2v, xf3v)

            elif self.sigma_model == "yearlycycle":
                raise ValueError(
                    f"bernoulli model {self.bernoulli_model} is not implemented yet"
                )
                sigma = l.yearlycycle(model, pm.Lognormal, "sigma", xf2v)

            elif self.sigma_model == "longterm_yearlycycle":
                # raise ValueError(f'bernoulli model {self.bernoulli_model} is not implemented yet')
                sigma = l.longterm_yearlycycle(
                    model,
                    pm.Lognormal("sigma_intercept", mu=0, sigma=1),
                    "sigma",
                    gmtv,
                    xf2v,
                )

            elif self.sigma_model == "longterm":
                # b_sigma is in the interval (0,inf)
                b_sigma = pm.Exponential("b_sigma", lam=1)
                # a_sigma is in the interval (-b, inf), mode at 0
                a_sigma = pm.Deterministic(
                    "a_sigma", tt.sub(pm.Exponential("as", lam=1), b_sigma)
                )
                sigma = pm.Deterministic(
                    "sigma", a_sigma * gmtv + b_sigma
                )  # in (0, inf)

            else:
                raise NotImplemented

            if not self.test:
                pm.Bernoulli(
                    "bernoulli", p=pbern, observed=df["is_dry_day"].astype(int)
                )
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
