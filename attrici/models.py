import numpy as np
import pymc as pm
import pytensor.tensor as tt
from scipy import stats

import attrici.distributions


class Pr(attrici.distributions.BernoulliGamma):
    """ Influence of GMT is modelled through a shift of
    mu and sigma parameters in a Beta distribution.
    """

    def __init__(self, modes):
        super(Pr, self).__init__()
        self.modes = modes
        self.test = False

    def setup(self, df_subset):
        model = pm.Model()

        with model:
            # so use df_subset directly.
            df_valid = df_subset.dropna(axis=0, how="any")

            gmt = pm.MutableData("gmt", df_subset["gmt_scaled"].values)
            xf0 = pm.MutableData("xf0", df_subset.filter(regex="^mode_0_").values)
            
            gmtv = pm.MutableData("gmtv", df_valid["gmt_scaled"].values)
            xf0_np = df_valid.filter(regex="^mode_0_").values
            xf0v = pm.MutableData("xf0v", xf0_np)

            covariates = pm.math.concatenate(
               [xf0, tt.tile(gmt[:, None], (1, int(xf0_np.shape[1]))) * xf0], axis=1
            )
            covariatesv = pm.math.concatenate(
                [xf0v, tt.tile(gmtv[:, None], (1, int(xf0_np.shape[1]))) * xf0v], axis=1
            )

            # pbern
            weights_pbern_longterm_intercept = pm.Normal(
                "weights_pbern_longterm_intercept", mu=0, sigma=1
            )
            weights_pbern_longterm_trend = pm.Normal(
                "weights_pbern_longterm_trend", mu=0, sigma=0.1
            )
            weights_pbern_fc_intercept = pm.math.concatenate(
                [
                    pm.Normal(
                        f"weights_pbern_fc_intercept_{i}",
                        mu=0,
                        sigma=1 / (2 * i + 1),
                        shape=2,
                    )
                    for i in range(int(xf0_np.shape[1]) // 2)
                ]
            )
            weights_pbern_fc_trend = pm.Normal(
                "weights_pbern_fc_trend", mu=0, sigma=0.1, shape=xf0_np.shape[1]
            )
            weights_pbern_fc = pm.math.concatenate(
                [weights_pbern_fc_intercept, weights_pbern_fc_trend]
            )
            logit_pbern = pm.Deterministic(
                "logit_pbern",
                tt.dot(covariates, weights_pbern_fc)
                + weights_pbern_longterm_intercept
                + weights_pbern_longterm_trend * gmt,
            )
            pbern = pm.Deterministic("pbern", pm.math.invlogit(logit_pbern))
            # mu
            weights_mu_longterm_intercept = pm.Normal(
                "weights_mu_longterm_intercept", mu=0, sigma=1
            )
            weights_mu_longterm_trend = pm.Normal(
                "weights_mu_longterm_trend", mu=0, sigma=0.1
            )
            weights_mu_fc_intercept = pm.math.concatenate(
                [
                    pm.Normal(
                        f"weights_mu_fc_intercept_{i}",
                        mu=0,
                        sigma=1 / (2 * i + 1),
                        shape=2,
                    )
                    for i in range(int(xf0_np.shape[1]) // 2)
                ]
            )
            weights_mu_fc_trend = pm.Normal(
                "weights_mu_fc_trend", mu=0, sigma=0.1, shape=xf0.shape[1]
            )
            weights_mu_fc = pm.math.concatenate(
                [weights_mu_fc_intercept, weights_mu_fc_trend]
            )
            eta_mu = (
                tt.dot(covariatesv, weights_mu_fc)
                + weights_mu_longterm_intercept
                + weights_mu_longterm_trend * gmtv
            )
            mu = pm.Deterministic("mu", pm.math.exp(eta_mu))
            # nu
            weights_nu_longterm_intercept = pm.Normal(
                "weights_nu_longterm_intercept", mu=0, sigma=1
            )
            weights_nu_fc_intercept = pm.math.concatenate(
                [
                    pm.Normal(
                        f"weights_nu_fc_intercept_{i}", mu=0, sigma=1 / (i + 1), shape=2
                    )
                    for i in range(int(xf0_np.shape[1]) // 2)
                ]
            )
            eta_nu = (
                tt.dot(xf0v, weights_nu_fc_intercept) + weights_nu_longterm_intercept
            )
            nu = pm.Deterministic("nu", pm.math.exp(eta_nu))
            sigma = pm.Deterministic("sigma", mu / nu)  # nu^2 = k -> k shape parameter

            logp_ = pm.Deterministic("logp", model.logp())

            if not self.test:
                pm.Bernoulli(
                    "bernoulli",
                    logit_p=logit_pbern,
                    observed=df_subset["is_dry_day"].astype(int),
                )
                pm.Gamma("obs", mu=mu, sigma=sigma, observed=df_valid["y_scaled"])
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

            df_valid = df_subset.dropna(axis=0, how="any")
            gmtv = pm.MutableData("gmt", df_valid["gmt_scaled"].values)
            xf0_np = df_valid.filter(regex="^mode_0_").values
            xf0 = pm.MutableData("xf0", xf0_np)
            print(xf0_np.shape)
            # mu

            weights_longterm_intercept = pm.Normal(
                "weights_longterm_intercept", mu=0, sigma=1
            )
            weights_longterm_trend = pm.Normal(
                "weights_longterm_trend", mu=0, sigma=0.1
            )
            weights_fc_intercept = pm.math.concatenate(
                [
                    pm.Normal(
                        f"weights_fc_intercept_{i}",
                        mu=0,
                        sigma=1 / (2 * i + 1),
                        shape=2,
                    )
                    for i in range(xf0_np.shape[1] // 2)
                ]
            )
            weights_fc_trend = pm.Normal(
                "weights_fc_trend", mu=0, sigma=0.1, shape=xf0_np.shape[1]
            )
            weights_fc = pm.math.concatenate([weights_fc_intercept, weights_fc_trend])
            # weights are the parameters that are learned by the model
            # eta is a linear model of the predictors
            eta = (
                tt.dot(
                    pm.math.concatenate(
                        [xf0, tt.tile(gmtv[:, None], (1, xf0_np.shape[1])) * xf0],
                        axis=1,
                    ),
                    weights_fc,
                )
                + weights_longterm_intercept
                + weights_longterm_trend * gmtv
            )
            mu = pm.Deterministic("mu", eta)

            # sigma
            weights_sigma_longterm_intercept = pm.Normal(
                "weights_sigma_longterm_intercept", mu=0, sigma=1
            )
            weights_sigma_fc_intercept = pm.math.concatenate(
                [
                    pm.Normal(
                        f"weights_sigma_fc_intercept_{i}",
                        mu=0,
                        sigma=1 / (2 * i + 1),
                        shape=2,
                    )
                    for i in range(int(xf0_np.shape[1]) // 2)
                ]
            )

            eta_sigma = (
                tt.dot(xf0, weights_sigma_fc_intercept)
                + weights_sigma_longterm_intercept
            )
            sigma = pm.Deterministic("sigma", pm.math.exp(eta_sigma))
            _ = pm.Deterministic("logp", model.logp())

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
            df_valid = df_subset.dropna(axis=0, how="any")
            gmtv = pm.Data("gmt", df_valid["gmt_scaled"].values)
            xf0 = pm.Data("xf0", df_valid.filter(regex="^mode_0_").values)
            # mu

            weights_longterm_intercept = pm.Normal(
                "weights_longterm_intercept", mu=0, sigma=1
            )
            weights_longterm_trend = pm.Normal(
                "weights_longterm_trend", mu=0, sigma=0.1
            )
            weights_fc_intercept = pm.math.concatenate(
                [
                    pm.Normal(
                        f"weights_fc_intercept_{i}",
                        mu=0,
                        sigma=1 / (2 * i + 1),
                        shape=2,
                    )
                    for i in range(int(xf0.shape[1]) // 2)
                ]
            )
            weights_fc_trend = pm.Normal(
                "weights_fc_trend", mu=0, sigma=0.1, shape=xf0.shape[1]
            )
            weights_fc = pm.math.concatenate([weights_fc_intercept, weights_fc_trend])
            # weights are the parameters that are learned by the model
            # eta is a linear model of the predictors
            eta = (
                tt.dot(
                    pm.math.concatenate(
                        [xf0, tt.tile(gmtv[:, None], (1, int(xf0.shape[1]))) * xf0],
                        axis=1,
                    ),
                    weights_fc,
                )
                + weights_longterm_intercept
                + weights_longterm_trend * gmtv
            )
            mu = pm.Deterministic("mu", eta)

            # sigma
            weights_sigma_longterm_intercept = pm.Normal(
                "weights_sigma_longterm_intercept", mu=0, sigma=1
            )
            weights_sigma_fc_intercept = pm.math.concatenate(
                [
                    pm.Normal(
                        f"weights_sigma_fc_intercept_{i}",
                        mu=0,
                        sigma=1 / (2 * i + 1),
                        shape=2,
                    )
                    for i in range(int(xf0.shape[1]) // 2)
                ]
            )

            eta_sigma = (
                tt.dot(xf0, weights_sigma_fc_intercept)
                + weights_sigma_longterm_intercept
            )
            sigma = pm.Deterministic("sigma", pm.math.exp(eta_sigma))
            logp_ = pm.Deterministic("logp", model.logpt)

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
            df_valid = df_subset.dropna(axis=0, how="any")
            gmtv = pm.Data("gmt", df_valid["gmt_scaled"].values)
            xf0 = pm.Data("xf0", df_valid.filter(regex="^mode_0_").values)
            # mu

            weights_longterm_intercept = pm.Normal(
                "weights_longterm_intercept", mu=0, sigma=1
            )
            weights_longterm_trend = pm.Normal(
                "weights_longterm_trend", mu=0, sigma=0.1
            )
            weights_fc_intercept = pm.math.concatenate(
                [
                    pm.Normal(
                        f"weights_fc_intercept_{i}",
                        mu=0,
                        sigma=1 / (2 * i + 1),
                        shape=2,
                    )
                    for i in range(int(xf0.shape[1]) // 2)
                ]
            )
            weights_fc_trend = pm.Normal(
                "weights_fc_trend", mu=0, sigma=0.1, shape=xf0.shape[1]
            )
            weights_fc = pm.math.concatenate([weights_fc_intercept, weights_fc_trend])
            # weights are the parameters that are learned by the model
            # eta is a linear model of the predictors
            eta = (
                tt.dot(
                    pm.math.concatenate(
                        [xf0, tt.tile(gmtv[:, None], (1, int(xf0.shape[1]))) * xf0],
                        axis=1,
                    ),
                    weights_fc,
                )
                + weights_longterm_intercept
                + weights_longterm_trend * gmtv
            )
            mu = pm.Deterministic("mu", eta)

            # sigma
            weights_sigma_longterm_intercept = pm.Normal(
                "weights_sigma_longterm_intercept", mu=0, sigma=1
            )
            weights_sigma_fc_intercept = pm.math.concatenate(
                [
                    pm.Normal(
                        f"weights_sigma_fc_intercept_{i}",
                        mu=0,
                        sigma=1 / (2 * i + 1),
                        shape=2,
                    )
                    for i in range(int(xf0.shape[1]) // 2)
                ]
            )

            eta_sigma = (
                tt.dot(xf0, weights_sigma_fc_intercept)
                + weights_sigma_longterm_intercept
            )
            sigma = pm.Deterministic("sigma", pm.math.exp(eta_sigma))
            logp_ = pm.Deterministic("logp", model.logpt)

            if not self.test:
                pm.Normal("obs", mu=mu, sigma=sigma, observed=df_valid["y_scaled"])

        return model


class Hurs(attrici.distributions.Beta):
    """ Influence of GMT on relative humidity (hurs) is modelled with Beta regression as proposed in
    https://www.tandfonline.com/doi/abs/10.1080/0266476042000214501
    """

    def __init__(self, modes):
        super(Hurs, self).__init__()
        self.modes = modes
        self.test = False

    def setup(self, df_subset):
        model = pm.Model()

        with model:
            # getting the predictors
            df_valid = df_subset.dropna(axis=0, how="any")
            gmtv = pm.MutableData("gmt", df_valid["gmt_scaled"].values)
            xf0_np = df_valid.filter(regex="^mode_0_").values
            xf0 = pm.MutableData("xf0", xf0_np)
            print(xf0_np.shape)          
            
            covariates = pm.math.concatenate(
                [xf0, tt.tile(gmtv[:, None], (1, int(xf0_np.shape[1]))) * xf0], axis=1
            )

            # phi is called the precision parameter
            weights_phi_longterm_intercept = pm.Normal(
                "weights_phi_longterm_intercept", mu=0, sigma=1
            )
            weights_phi_fc_intercept = pm.math.concatenate(
                [
                    pm.Normal(
                        f"weights_phi_fc_intercept_{i}",
                        mu=0,
                        sigma=1 / (2 * i + 1),
                        shape=2,
                    )
                    for i in range(int(xf0_np.shape[1]) // 2)
                ]
            )

            eta_phi = (
                tt.dot(xf0, weights_phi_fc_intercept) + weights_phi_longterm_intercept
            )
            phi = pm.math.exp(eta_phi)

            weights_longterm_intercept = pm.Normal(
                "weights_longterm_intercept", mu=0, sigma=1
            )
            weights_longterm_trend = pm.Normal(
                "weights_longterm_trend", mu=0, sigma=0.1
            )
            weights_fc_intercept = pm.math.concatenate(
                [
                    pm.Normal(
                        f"weights_fc_intercept_{i}",
                        mu=0,
                        sigma=1 / (2 * i + 1),
                        shape=2,
                    )
                    for i in range(int(xf0_np.shape[1]) // 2)
                ]
            )
            weights_fc_trend = pm.Normal(
                "weights_fc_trend", mu=0, sigma=0.1, shape=xf0_np.shape[1]
            )
            weights_fc = pm.math.concatenate([weights_fc_intercept, weights_fc_trend])
            # weights are the parameters that are learned by the model
            # eta is a linear model of the predictors
            eta = (
                tt.dot(covariates, weights_fc)
                + weights_longterm_intercept
                + weights_longterm_trend * gmtv
            )

            # mu models the expected value of the target data.
            # The logit function is the link function in the Generalized Linear Model
            mu = pm.math.invlogit(eta)

            alpha = pm.Deterministic("alpha", mu * phi)

            beta = pm.Deterministic("beta", (1 - mu) * phi)
            logp_ = pm.Deterministic("logp", model.logp())

            if not self.test:
                pm.Beta("obs", alpha=alpha, beta=beta, observed=df_valid["y_scaled"])

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
            gmtv = pm.MutableData("gmt", df_valid["gmt_scaled"].values)
            xf0_np = df_valid.filter(regex="^mode_0_").values
            xf0 = pm.MutableData("xf0", xf0_np)
            print(xf0_np.shape)
            # mu

            weights_longterm_intercept = pm.Normal(
                "weights_longterm_intercept", mu=0, sigma=1
            )
            weights_longterm_trend = pm.Normal(
                "weights_longterm_trend", mu=0, sigma=0.1
            )
            weights_fc_intercept = pm.math.concatenate(
                [
                    pm.Normal(
                        f"weights_fc_intercept_{i}",
                        mu=0,
                        sigma=1 / (2 * i + 1),
                        shape=2,
                    )
                    for i in range(int(xf0_np.shape[1]) // 2)
                ]
            )
            weights_fc_trend = pm.Normal(
                "weights_fc_trend", mu=0, sigma=0.1, shape=xf0_np.shape[1]
            )
            weights_fc = pm.math.concatenate([weights_fc_intercept, weights_fc_trend])
            # weights are the parameters that are learned by the model
            # eta is a linear model of the predictors
            eta = (
                tt.dot(
                    pm.math.concatenate(
                        [xf0, tt.tile(gmtv[:, None], (1, int(xf0_np.shape[1]))) * xf0],
                        axis=1,
                    ),
                    weights_fc,
                )
                + weights_longterm_intercept
                + weights_longterm_trend * gmtv
            )
            mu = pm.Deterministic("mu", eta)

            # sigma
            weights_sigma_longterm_intercept = pm.Normal(
                "weights_sigma_longterm_intercept", mu=0, sigma=1
            )
            weights_sigma_fc_intercept = pm.math.concatenate(
                [
                    pm.Normal(
                        f"weights_sigma_fc_intercept_{i}",
                        mu=0,
                        sigma=1 / (2 * i + 1),
                        shape=2,
                    )
                    for i in range(int(xf0_np.shape[1]) // 2)
                ]
            )

            eta_sigma = (
                tt.dot(xf0, weights_sigma_fc_intercept)
                + weights_sigma_longterm_intercept
            )
            sigma = pm.Deterministic("sigma", pm.math.exp(eta_sigma))
            _ = pm.Deterministic("logp", model.logp())

            if not self.test:
                pm.Normal("obs", mu=mu, sigma=sigma, observed=df_valid["y_scaled"])

        return model


class Rsds(attrici.distributions.Normal):
    """ Influence of GMT is modelled through a shift of
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

            df_valid = df_subset.dropna(axis=0, how="any")
            gmtv = pm.MutableData("gmt", df_valid["gmt_scaled"].values)
            xf0_np = df_valid.filter(regex="^mode_0_").values
            xf0 = pm.MutableData("xf0", xf0_np)
            print(xf0_np.shape)
            # mu

            weights_longterm_intercept = pm.Normal(
                "weights_longterm_intercept", mu=0, sigma=1
            )
            weights_longterm_trend = pm.Normal(
                "weights_longterm_trend", mu=0, sigma=0.1
            )
            weights_fc_intercept = pm.math.concatenate(
                [
                    pm.Normal(
                        f"weights_fc_intercept_{i}",
                        mu=0,
                        sigma=1 / (2 * i + 1),
                        shape=2,
                    )
                    for i in range(int(xf0_np.shape[1]) // 2)
                ]
            )
            weights_fc_trend = pm.Normal(
                "weights_fc_trend", mu=0, sigma=0.1, shape=xf0_np.shape[1]
            )
            weights_fc = pm.math.concatenate([weights_fc_intercept, weights_fc_trend])
            # weights are the parameters that are learned by the model
            # eta is a linear model of the predictors
            eta = (
                tt.dot(
                    pm.math.concatenate(
                        [xf0, tt.tile(gmtv[:, None], (1, int(xf0_np.shape[1]))) * xf0],
                        axis=1,
                    ),
                    weights_fc,
                )
                + weights_longterm_intercept
                + weights_longterm_trend * gmtv
            )
            mu = pm.Deterministic("mu", eta)

            # sigma
            weights_sigma_longterm_intercept = pm.Normal(
                "weights_sigma_longterm_intercept", mu=0, sigma=1
            )
            weights_sigma_fc_intercept = pm.math.concatenate(
                [
                    pm.Normal(
                        f"weights_sigma_fc_intercept_{i}",
                        mu=0,
                        sigma=1 / (2 * i + 1),
                        shape=2,
                    )
                    for i in range(int(xf0_np.shape[1]) // 2)
                ]
            )

            eta_sigma = (
                tt.dot(xf0, weights_sigma_fc_intercept)
                + weights_sigma_longterm_intercept
            )
            sigma = pm.Deterministic("sigma", pm.math.exp(eta_sigma))
            logp_ = pm.Deterministic("logp", model.logp())

            if not self.test:
                pm.Normal("obs", mu=mu, sigma=sigma, observed=df_valid["y_scaled"])

        return model


class RsdsWeibull(attrici.distributions.Weibull):
    """ Influence of GMT is modelled through a shift of
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
            df_valid = df_subset.dropna(axis=0, how="any")
            gmtv = pm.Data("gmt", df_valid["gmt_scaled"].values)
            xf0 = pm.Data("xf0", df_valid.filter(regex="^mode_0_").values)

            # beta
            weights_longterm_intercept = pm.Normal(
                "weights_longterm_intercept", mu=0, sigma=1
            )
            weights_longterm_trend = pm.Normal(
                "weights_longterm_trend", mu=0, sigma=0.1
            )
            weights_fc_intercept = pm.math.concatenate(
                [
                    pm.Normal(
                        f"weights_fc_intercept_{i}",
                        mu=0,
                        sigma=1 / (2 * i + 1),
                        shape=2,
                    )
                    for i in range(int(xf0.shape[1]) // 2)
                ]
            )
            weights_fc_trend = pm.Normal(
                "weights_fc_trend", mu=0, sigma=0.1, shape=xf0.shape[1]
            )
            weights_fc = pm.math.concatenate([weights_fc_intercept, weights_fc_trend])
            # weights are the parameters that are learned by the model
            # eta is a linear model of the predictors
            eta = (
                tt.dot(
                    pm.math.concatenate(
                        [xf0, tt.tile(gmtv[:, None], (1, int(xf0.shape[1]))) * xf0],
                        axis=1,
                    ),
                    weights_fc,
                )
                + weights_longterm_intercept
                + weights_longterm_trend * gmtv
            )
            beta = pm.Deterministic("beta", pm.math.exp(eta))

            # alpha
            weights_alpha_longterm_intercept = pm.Normal(
                "weights_alpha_longterm_intercept", mu=0, sigma=1
            )
            weights_alpha_fc_intercept = pm.math.concatenate(
                [
                    pm.Normal(
                        f"weights_alpha_fc_intercept_{i}",
                        mu=0,
                        sigma=1 / (2 * i + 1),
                        shape=2,
                    )
                    for i in range(int(xf0.shape[1]) // 2)
                ]
            )

            eta_alpha = (
                tt.dot(xf0, weights_alpha_fc_intercept)
                + weights_alpha_longterm_intercept
            )
            
            alpha = pm.Deterministic("alpha", pm.math.exp(eta_alpha))
            logp_ = pm.Deterministic("logp", model.logpt)

            if not self.test:
                pm.Weibull("obs", alpha=alpha, beta=beta, observed=df_valid["y_scaled"])

        return model


class Tasrange(attrici.distributions.Gamma):
    """ Influence of GMT is modelled through a shift of
    mu and sigma parameters in a Beta distribution.
    """

    def __init__(self, modes):
        super(Tasrange, self).__init__()
        self.modes = modes
        self.test = False

    def setup(self, df_subset):
        model = pm.Model()

        with model:
            # so use df_subset directly.
            df_valid = df_subset.dropna(axis=0, how="any")
            gmtv = pm.MutableData("gmt", df_valid["gmt_scaled"].values)
            xf0_np = df_valid.filter(regex="^mode_0_").values
            xf0 = pm.MutableData("xf0", xf0_np)
            print("xfo shape" ,xf0.shape)
            print("xfo_np shape", xf0_np.shape)

            covariates = pm.math.concatenate(
                [xf0, tt.tile(gmtv[:, None], (1, int(xf0_np.shape[1]))) * xf0], axis=1
            )

            weights_nu_longterm_intercept = pm.Normal(
                "weights_nu_longterm_intercept", mu=0, sigma=1
            )
            weights_nu_fc_intercept = pm.math.concatenate(
                [
                    pm.Normal(
                        f"weights_nu_fc_intercept_{i}", mu=0, sigma=1 / (i + 1), shape=2
                    )
                    for i in range(int(xf0_np.shape[1]) // 2)
                ]
            )

            eta_nu = (
                tt.dot(xf0, weights_nu_fc_intercept) + weights_nu_longterm_intercept
            )
            nu = pm.math.exp(eta_nu)

            weights_longterm_intercept = pm.Normal(
                "weights_longterm_intercept", mu=0, sigma=1
            )
            weights_longterm_trend = pm.Normal(
                "weights_longterm_trend", mu=0, sigma=0.1
            )
            weights_fc_intercept = pm.math.concatenate(
                [
                    pm.Normal(
                        f"weights_fc_intercept_{i}",
                        mu=0,
                        sigma=1 / (2 * i + 1),
                        shape=2,
                    )
                    for i in range(int(xf0_np.shape[1]) // 2)
                ]
            )
            weights_fc_trend = pm.Normal(
                "weights_fc_trend", mu=0, sigma=0.1, shape=xf0_np.shape[1]
            )
            weights_fc = pm.math.concatenate([weights_fc_intercept, weights_fc_trend])
            # weights are the parameters that are learned by the model
            # eta is a linear model of the predictors
            eta = (
                tt.dot(covariates, weights_fc)
                + weights_longterm_intercept
                + weights_longterm_trend * gmtv
            )

            # mu models the expected value of the target data.
            # The logit function is the link function in the Generalized Linear Model
            mu = pm.Deterministic("mu", pm.math.exp(eta))
            sigma = pm.Deterministic("sigma", mu / nu)
            logp_ = pm.Deterministic("logp", model.logp())

            if not self.test:
                pm.Gamma("obs", mu=mu, sigma=sigma, observed=df_valid["y_scaled"])

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
            df_valid = df_subset.dropna(axis=0, how="any")
            gmtv = pm.MutableData("gmt", df_valid["gmt_scaled"].values)
            xf0_np = df_valid.filter(regex="^mode_0_").values
            xf0 = pm.MutableData("xf0", xf0_np)
            print(xf0_np.shape)
            # beta

            weights_longterm_intercept = pm.Normal(
                "weights_longterm_intercept", mu=0, sigma=1
            )
            weights_longterm_trend = pm.Normal(
                "weights_longterm_trend", mu=0, sigma=0.1
            )
            weights_fc_intercept = pm.math.concatenate(
                [
                    pm.Normal(
                        f"weights_fc_intercept_{i}",
                        mu=0,
                        sigma=1 / (2 * i + 1),
                        shape=2,
                    )
                    for i in range(int(xf0_np.shape[1]) // 2)
                ]
            )
            weights_fc_trend = pm.Normal(
                "weights_fc_trend", mu=0, sigma=0.1, shape=xf0_np.shape[1]
            )
            weights_fc = pm.math.concatenate([weights_fc_intercept, weights_fc_trend])
            # weights are the parameters that are learned by the model
            # eta is a linear model of the predictors
            eta = (
                tt.dot(
                    pm.math.concatenate(
                        [xf0, tt.tile(gmtv[:, None], (1, int(xf0_np.shape[1]))) * xf0],
                        axis=1,
                    ),
                    weights_fc,
                )
                + weights_longterm_intercept
                + weights_longterm_trend * gmtv
            )
            beta = pm.Deterministic("beta", pm.math.exp(eta))

            # alpha
            weights_alpha_longterm_intercept = pm.Normal(
                "weights_alpha_longterm_intercept", mu=0, sigma=1
            )
            weights_alpha_fc_intercept = pm.math.concatenate(
                [
                    pm.Normal(
                        f"weights_alpha_fc_intercept_{i}",
                        mu=0,
                        sigma=1 / (2 * i + 1),
                        shape=2,
                    )
                    for i in range(int(xf0_np.shape[1]) // 2)
                ]
            )

            eta_alpha = (
                tt.dot(xf0, weights_alpha_fc_intercept)
                + weights_alpha_longterm_intercept
            )
            alpha = pm.Deterministic("alpha", pm.math.exp(eta_alpha))
            logp_ = pm.Deterministic("logp", model.logp())

            # mu = pm.math.invlogit(eta)
            # alpha = pm.Deterministic("alpha", mu * phi)
            # beta = pm.Deterministic("beta", (1 - mu) * phi)

            if not self.test:
                pm.Weibull("obs", alpha=alpha, beta=beta, observed=df_valid["y_scaled"])

        return model