import numpy as np
from scipy import stats

class Distribution(object):

    pass


class Normal(Distribution):

    def __init__(self):

        self.parameter_bounds = {"mu":[None,None], "sigma":[0,None]}

        print("Using Normal distribution model.")

    def quantile_mapping(self, d, y_scaled):

        """
        specific for normally distributed variables.
        """
        quantile = stats.norm.cdf(y_scaled, loc=d["mu"], scale=d["sigma"])
        x_mapped = stats.norm.ppf(quantile, loc=d["mu_ref"], scale=d["sigma_ref"])

        return x_mapped


class BernoulliGamma(Distribution):

    def __init__(self):

        self.parameter_bounds = {"pbern":[0,1], "mu":[0,None], "sigma":[0,None]}

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

class Beta(Distribution):

    pass


class Rice(Distribution):

    pass

class Weibull(Distribution):

    pass


