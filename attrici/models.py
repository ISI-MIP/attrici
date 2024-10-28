"""
Models.
"""

import numpy as np
from scipy import stats

import attrici.distributions
from attrici.modelling import pm, tt


def filter_df_valid(df):
    return df.dropna(axis=0, how="any")


def get_gmt_xf0(df):
    gmt = pm.Data("gmt", df["gmt_scaled"].values)
    xf0 = pm.Data("xf0", df.filter(regex="^mode_0_").values)
    return gmt, xf0


def calc_linear_model(name, predictor, predictor_trend=None):
    """Calculate a linear model.

    Parameters
    ----------
    name : str
      Internal name of variable, e.g. sigma, phi
    predictor : np.ndarray
      Values, e.g. `xf0`
    predictor_trend : np.ndarray
      Values, e.g. `gmt`. When `None` only linear model without trend is calculated.

    Returns
    -------
    Theano tensor
      Linear model

    """

    weights_longterm_intercept = pm.Normal(
        f"weights_{name}_longterm_intercept", mu=0, sd=1
    )
    weights_fc_intercept = pm.math.concatenate(
        [
            pm.Normal(
                f"weights_{name}_fc_intercept_{i}",
                mu=0,
                sd=1 / (2 * i + 1),
                shape=2,
            )
            for i in range(int(predictor.dshape[1]) // 2)
        ]
    )
    if predictor_trend is None:
        return tt.dot(predictor, weights_fc_intercept) + weights_longterm_intercept

    covariates = pm.math.concatenate(
        [
            predictor,
            tt.tile(predictor_trend[:, None], (1, int(predictor.dshape[1])))
            * predictor,
        ],
        axis=1,
    )
    weights_longterm_trend = pm.Normal(f"weights_{name}_longterm_trend", mu=0, sd=0.1)
    weights_fc_trend = pm.Normal(
        f"weights_{name}_fc_trend", mu=0, sd=0.1, shape=predictor.dshape[1]
    )
    weights_fc = pm.math.concatenate([weights_fc_intercept, weights_fc_trend])
    return (
        tt.dot(covariates, weights_fc)
        + weights_longterm_intercept
        + weights_longterm_trend * predictor_trend
    )


class Pr(attrici.distributions.BernoulliGamma):
    """Influence of GMT is modelled through a shift of
    mu and sigma parameters in a Beta distribution.
    """

    def __init__(self, modes):
        super(Pr, self).__init__()
        self.modes = modes
        self.test = False

    def setup(self, df_subset):
        model = pm.Model()

        with model:
            # use df_subset directly
            gmt, xf0 = get_gmt_xf0(df_subset)
            logit_pbern = calc_linear_model("pbern", gmt, xf0)

            df_valid = filter_df_valid(df_subset)
            gmtv, xf0v = get_gmt_xf0(df_valid)
            mu = pm.Deterministic(
                "mu", pm.math.exp(calc_linear_model("mu", gmtv, xf0v))
            )
            nu = pm.Deterministic("nu", pm.math.exp(calc_linear_model("nu", xf0v)))
            sigma = pm.Deterministic("sigma", mu / nu)  # nu^2 = k -> k shape parameter

            pm.Deterministic("pbern", pm.math.invlogit(logit_pbern))
            pm.Deterministic("logp", model.logpt)

            if not self.test:
                pm.Bernoulli(
                    "bernoulli",
                    logit_p=pm.Deterministic("logit_pbern", logit_pbern),
                    observed=df_subset["is_dry_day"].astype(int),
                )
                pm.Gamma("obs", mu=mu, sigma=sigma, observed=df_valid["y_scaled"])
        return model


class Tas(attrici.distributions.Normal):
    """Influence of GMT is modelled through a shift of
    mu parameter in the Normal distribution.
    """

    def __init__(self, modes):
        super(Tas, self).__init__()
        self.modes = modes
        self.test = False

    def setup(self, df_subset):
        model = pm.Model()

        with model:
            df_valid = filter_df_valid(df_subset)
            gmt, xf0 = get_gmt_xf0(df_valid)

            mu = pm.Deterministic("mu", calc_linear_model("mu", xf0, gmt))
            sigma = pm.Deterministic(
                "sigma", pm.math.exp(calc_linear_model("sigma", xf0))
            )
            pm.Deterministic("logp", model.logpt)

            if not self.test:
                pm.Normal("obs", mu=mu, sigma=sigma, observed=df_valid["y_scaled"])

        return model


class Rlds(attrici.distributions.Normal):
    """Influence of GMT on longwave downwelling shortwave radiation
    is modelled through a shift of mu parameter in the Normal distribution.
    """

    def __init__(self, modes):
        super(Rlds, self).__init__()
        self.modes = modes
        self.test = False

    def setup(self, df_subset):
        model = pm.Model()

        with model:
            df_valid = filter_df_valid(df_subset)
            gmt, xf0 = get_gmt_xf0(df_valid)

            mu = pm.Deterministic("mu", calc_linear_model("mu", xf0, gmt))
            sigma = pm.Deterministic(
                "sigma", pm.math.exp("sigma", calc_linear_model(xf0))
            )
            pm.Deterministic("logp", model.logpt)

            if not self.test:
                pm.Normal("obs", mu=mu, sigma=sigma, observed=df_valid["y_scaled"])

        return model


class Ps(attrici.distributions.Normal):
    """Influence of GMT on sea level pressure (ps) is modelled through a shift of
    mu parameter in the Normal distribution.
    """

    def __init__(self, modes):
        super(Ps, self).__init__()
        self.modes = modes
        self.test = False

    def setup(self, df_subset):
        model = pm.Model()

        with model:
            df_valid = filter_df_valid(df_subset)
            gmt, xf0 = get_gmt_xf0(df_valid)

            mu = pm.Deterministic("mu", calc_linear_model("mu", xf0, gmt))
            sigma = pm.Deterministic(
                "sigma", pm.math.exp(calc_linear_model("sigma", xf0))
            )
            pm.Deterministic("logp", model.logpt)

            if not self.test:
                pm.Normal("obs", mu=mu, sigma=sigma, observed=df_valid["y_scaled"])

        return model


class Hurs(attrici.distributions.Beta):
    """Influence of GMT on relative humidity (hurs) is modelled with Beta
    regression as proposed in
    https://www.tandfonline.com/doi/abs/10.1080/0266476042000214501
    """

    def __init__(self, modes):
        super(Hurs, self).__init__()
        self.modes = modes
        self.test = False

    def setup(self, df_subset):
        model = pm.Model()

        with model:
            df_valid = filter_df_valid(df_subset)
            gmt, xf0 = get_gmt_xf0(df_valid)

            phi = pm.math.exp(calc_linear_model("phi", xf0))

            # mu models the expected value of the target data.
            # The logit function is the link function in the Generalized Linear Model
            mu = pm.math.invlogit(calc_linear_model("mu", xf0, gmt))

            alpha = pm.Deterministic("alpha", mu * phi)
            beta = pm.Deterministic("beta", (1 - mu) * phi)
            pm.Deterministic("logp", model.logpt)

            if not self.test:
                pm.Beta("obs", alpha=alpha, beta=beta, observed=df_valid["y_scaled"])

        return model


class Tasskew(attrici.distributions.Normal):
    """Influence of GMT is modelled through a shift of
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
            df_valid = filter_df_valid(df_subset)
            gmt, xf0 = get_gmt_xf0(df_valid)

            mu = pm.Deterministic("mu", calc_linear_model("mu", xf0, gmt))
            sigma = pm.Deterministic(
                "sigma", pm.math.exp(calc_linear_model("sigma", xf0))
            )
            pm.Deterministic("logp", model.logpt)

            if not self.test:
                pm.Normal("obs", mu=mu, sigma=sigma, observed=df_valid["y_scaled"])

        return model


class Rsds(attrici.distributions.Normal):
    """Influence of GMT is modelled through a shift of
    mu and sigma parameters in a Normal distribution.
    """

    def __init__(self, modes):
        super(Rsds, self).__init__()
        self.modes = modes
        self.test = False

    def quantile_mapping(self, d, y_scaled):
        """
        nan values are not quantile-mapped. 0 rsds happens mainly in the polar night.
        """

        quantile = stats.norm.cdf(y_scaled, loc=d["mu"], scale=d["sigma"])
        x_mapped = stats.norm.ppf(quantile, loc=d["mu_ref"], scale=d["sigma_ref"])

        x_mapped[x_mapped <= 0] = np.nan

        return x_mapped

    def setup(self, df_subset):
        model = pm.Model()

        with model:
            df_valid = filter_df_valid(df_subset)
            gmt, xf0 = get_gmt_xf0(df_valid)

            mu = pm.Deterministic("mu", calc_linear_model("mu", xf0, gmt))
            sigma = pm.Deterministic(
                "sigma", pm.math.exp(calc_linear_model("sigma", xf0))
            )
            pm.Deterministic("logp", model.logpt)

            if not self.test:
                pm.Normal("obs", mu=mu, sigma=sigma, observed=df_valid["y_scaled"])

        return model


class RsdsWeibull(attrici.distributions.Weibull):
    """Influence of GMT is modelled through a shift of
    the scale parameter beta in the Weibull distribution. The shape
    parameter alpha is assumed free of a trend.

    """

    def __init__(self, modes):
        super(RsdsWeibull, self).__init__()
        self.modes = modes
        self.test = False

    def setup(self, df_subset):
        model = pm.Model()

        with model:
            df_valid = filter_df_valid(df_subset)
            gmt, xf0 = get_gmt_xf0(df_valid)

            beta = pm.Deterministic(
                "beta", pm.math.exp(calc_linear_model("beta", xf0, gmt))
            )
            alpha = pm.Deterministic(
                "alpha", pm.math.exp(calc_linear_model("alpha", xf0))
            )
            pm.Deterministic("logp", model.logpt)

            if not self.test:
                pm.Weibull("obs", alpha=alpha, beta=beta, observed=df_valid["y_scaled"])

        return model


class Tasrange(attrici.distributions.Gamma):
    """Influence of GMT is modelled through a shift of
    mu and sigma parameters in a Beta distribution.
    """

    def __init__(self, modes):
        super(Tasrange, self).__init__()
        self.modes = modes
        self.test = False

    def setup(self, df_subset):
        model = pm.Model()

        with model:
            df_valid = filter_df_valid(df_subset)
            gmt, xf0 = get_gmt_xf0(df_valid)

            nu = pm.math.exp(calc_linear_model("nu", xf0))
            # mu models the expected value of the target data.
            # The logit function is the link function in the Generalized Linear Model
            mu = pm.Deterministic("mu", pm.math.exp(calc_linear_model("mu", xf0, gmt)))
            sigma = pm.Deterministic("sigma", mu / nu)
            pm.Deterministic("logp", model.logpt)

            if not self.test:
                pm.Gamma("obs", mu=mu, sigma=sigma, observed=df_valid["y_scaled"])

        return model


class Wind(attrici.distributions.Weibull):
    """Influence of GMT is modelled through a shift of
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
            df_valid = filter_df_valid(df_subset)
            gmt, xf0 = get_gmt_xf0(df_valid)

            beta = pm.Deterministic(
                "beta", pm.math.exp(calc_linear_model("beta", xf0, gmt))
            )
            alpha = pm.Deterministic(
                "alpha", pm.math.exp(calc_linear_model("alpha", xf0))
            )
            pm.Deterministic("logp", model.logpt)

            if not self.test:
                pm.Weibull("obs", alpha=alpha, beta=beta, observed=df_valid["y_scaled"])

        return model
