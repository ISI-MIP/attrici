"""
Distributions.
"""

import numpy as np
from loguru import logger
from scipy import stats

from attrici.modelling import pm


class Distribution:
    def __init__(self):
        logger.info("Using {} distribution model.", type(self).__name__)

    def resample_missing(self, trace, df, model, progressbar):
        # TODO this breaks if first parameter does not have time dimension
        # but second parameter has. It therefore requires an order in self.params
        logger.warning("Trace is not complete due to masked data. Resample missing.")

        with model:
            # use all data for the model specific data-inputs
            # if input is available in the model
            input_vars = {"gmt": "gmt_scaled", "gmtv": "gmt_scaled"}
            fourier_vars = {
                "xf0": "^mode_0_",
                "xf0v": "^mode_0_",
                "xf1": "^mode_1_",
                "xf2": "^mode_2_",
                "xf3": "^mode_3_",
                "posxf0": "posmode_0_",
            }
            for key, df_key in input_vars.items():
                try:
                    pm.set_data({key: df[df_key].values})
                    logger.info("Replaced {} in model with full data-set", key)
                except KeyError:
                    pass

            for key, df_key in fourier_vars.items():
                try:
                    pm.set_data({key: df.filter(regex=df_key).values})
                    logger.info("Replaced {} in model with full data-set", key)
                except KeyError:
                    pass

            trace_obs = pm.sample_posterior_predictive(
                [trace],
                samples=1,
                var_names=self.params + ["logp"],  # + ["obs"],
                progressbar=progressbar,
            )
            for gmt in ["gmt", "gmtv"]:
                try:
                    pm.set_data({gmt: np.zeros_like(df["gmt_scaled"])})
                except KeyError:
                    pass
            trace_cfact = pm.sample_posterior_predictive(
                [trace],
                samples=1,
                var_names=self.params + ["logp"],  # + ["obs"],
                progressbar=progressbar,
            )
        logger.info("Resampled missing.")
        return trace_obs, trace_cfact


class Normal(Distribution):
    def __init__(self):
        super(Normal, self).__init__()
        self.params = ["mu", "sigma"]
        self.parameter_bounds = {"mu": [None, None], "sigma": [0, None]}

    def quantile_mapping(self, d, y_scaled):
        """
        specific for normally distributed variables.
        """
        quantile = stats.norm.cdf(y_scaled, loc=d["mu"], scale=d["sigma"])
        x_mapped = stats.norm.ppf(quantile, loc=d["mu_ref"], scale=d["sigma_ref"])

        return x_mapped


class BernoulliGamma(Distribution):
    def __init__(self):
        super(BernoulliGamma, self).__init__()
        self.params = ["mu", "sigma", "pbern"]
        self.parameter_bounds = {"pbern": [0, 1], "mu": [0, None], "sigma": [0, None]}

    def quantile_mapping(self, d, y_scaled):
        """Needs a thorough description of QM for BernoulliGamma"""

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
        # TODO have this zero precip at dry day fix earlier (in const.py for example)
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
        logger.info(
            "Normal qm for higher cfact dry probability: {}", do_normal_qm_0.sum()
        )
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
        logger.info(
            "Normal qm for higher cfact wet probability: {}", do_normal_qm_1.sum()
        )
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
        logger.info("Days originally dry: {}", dry_day.sum())
        logger.info("Days made wet: {}", randomly_map_to_wet.sum())
        logger.info("Days in cfact dry: {}", (cfact == 0).sum())
        logger.info(
            "Total days: {}",
            (cfact == 0).sum()
            + randomly_map_to_wet.sum()
            + do_normal_qm_0.sum()
            + do_normal_qm_1.sum(),
        )

        return cfact


class Gamma(Distribution):
    def __init__(self):
        super(Gamma, self).__init__()
        self.params = ["mu", "sigma"]
        self.parameter_bounds = {"mu": [0, None], "sigma": [0, None]}

    def quantile_mapping(self, d, y_scaled):
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


class Beta(Distribution):
    def __init__(self):
        super(Beta, self).__init__()
        self.params = ["alpha", "beta"]
        self.parameter_bounds = {"alpha": [0, None], "beta": [0, None]}

    def quantile_mapping(self, d, y_scaled):
        quantile = stats.beta.cdf(y_scaled, d["alpha"], d["beta"])
        x_mapped = stats.beta.ppf(quantile, d["alpha_ref"], d["beta_ref"])

        return x_mapped


class Rice(Distribution):
    def __init__(self):
        super(Rice, self).__init__()
        self.params = ["nu", "sigma"]
        self.parameter_bounds = {"nu": [0, None], "sigma": [0, None]}

    def quantile_mapping(self, d, y_scaled):
        quantile = stats.rice.cdf(y_scaled, b=d["nu"] / d["sigma"], scale=d["sigma"])
        x_mapped = stats.rice.ppf(
            quantile, b=d["nu_ref"] / d["sigma_ref"], scale=d["sigma_ref"]
        )

        return x_mapped


class Weibull(Distribution):
    def __init__(self):
        super(Weibull, self).__init__()
        # ensure that parameter with time dimension is first
        self.params = ["beta", "alpha"]
        self.parameter_bounds = {"alpha": [0, None], "beta": [0, None]}

    def quantile_mapping(self, d, y_scaled):
        quantile = stats.weibull_min.cdf(y_scaled, d["alpha"], scale=d["beta"])
        x_mapped = stats.weibull_min.ppf(quantile, d["alpha_ref"], scale=d["beta_ref"])

        return x_mapped
