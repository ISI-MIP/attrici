import numpy as np
import pymc3 as pm
from scipy import stats
import pandas as pd
import theano.tensor as tt
import icounter.distributions

class PrecipitationLongterm(icounter.distributions.BernoulliGamma):

    """ Influence of GMT is modelled through the parameters of the Gamma
    distribution. Example: precipitation """

    def __init__(self, modes):

        super(PrecipitationLongterm, self).__init__()
        self.modes = modes
        self.mu_model = mu_model
        self.sigma_model = sigma_model
        self.bernoulli_model = bernoulli_model
        self.test = False

        print("Using PrecipitationLongterm distribution model.")

    def setup(self, df_valid, df):

        model = pm.Model()

        with model:

            gmt = pm.Data("gmt", df["gmt_scaled"].values)
            # FIXME: avoid this doubling of predictors, introduced for BernoulliGamma
            gmtv = pm.Data("gmtv", df_valid["gmt_scaled"].values)

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
            )

            # b_mu is in the interval (0,inf)
            b_mu = pm.Exponential("b_mu", lam=1)
            # a_mu in (-b, inf)
            a_mu = pm.Deterministic("a_mu", pm.Exponential("am", lam=1) - b_mu)
            mu = pm.Deterministic("mu", a_mu * gmtv + b_mu)  # in (0, inf)

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
                    "bernoulli", p=pbern, observed=df["is_dry_day"].astype(int)
                )
                pm.Gamma("obs", mu=mu, sigma=sigma, observed=df_valid["y_scaled"])

        return model

