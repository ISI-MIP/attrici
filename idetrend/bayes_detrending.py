import os
import theano
import numpy as np
import pymc3 as pm
import matplotlib.pylab as plt
import pandas as pd
import netCDF4 as nc
from datetime import datetime
from pathlib import Path
import idetrend.datahandler as dh


def y_norm(y_to_scale, y_orig):
    return (y_to_scale - y_orig.min()) / (y_orig.max() - y_orig.min())


def y_inv(y, y_orig):
    """rescale data y to y_original"""
    return y * (y_orig.max() - y_orig.min()) + y_orig.min()


def rescale(y, y_orig):
    """rescale data y to y_original"""
    return y * (y_orig.max() - y_orig.min())


class bayes_regression(object):
    def __init__(self, cfg):

        self.output_dir = cfg.output_dir
        self.init = cfg.init
        self.draws = cfg.draws
        self.cores = cfg.ncores_per_job
        self.chains = cfg.chains
        self.tune = cfg.tune
        self.subset = cfg.subset

        self.progressbar = cfg.progressbar
        self.days_of_year = cfg.days_of_year

        self.modes = cfg.modes
        self.linear_mu = cfg.linear_mu
        self.linear_sigma = cfg.linear_sigma
        self.sigma_beta = cfg.sigma_beta
        self.smu = cfg.smu
        self.sps = cfg.sps
        self.stmu = cfg.stmu
        self.stps = cfg.stps

    def setup_model(self, df):

        # create instance of pymc model class
        df_subset = df.loc[::self.subset].copy()
        regressor = df_subset["gmt_scaled"].values
        x_fourier = rescale_fourier(df_subset, self.modes)
        self.model = pm.Model()

        with self.model:
            slope = pm.Normal("slope", self.linear_mu, self.linear_sigma)
            intercept = pm.Normal("intercept", self.linear_mu, self.linear_sigma)
            sigma = pm.HalfCauchy("sigma", self.sigma_beta, testval=1)

            beta_yearly = pm.Normal("beta_yearly", mu=self.smu, sd=self.sps, shape=2 * self.modes)
            beta_trend = pm.Normal("beta_trend", mu=self.stmu, sd=self.stps, shape=2 * self.modes)

            estimated = (
                intercept
                + slope * regressor
                + det_dot(x_fourier, beta_yearly)
                + (regressor * det_dot(x_fourier, beta_trend))
            )
            out = pm.Normal(
                "obs", mu=estimated, sd=sigma, observed=df_subset["y_scaled"]
            )

        self.df = df
        return self.model, x_fourier

    def run(self, df, lat, lon):

        self.setup_model(df)

        outdir_for_cell = dh.make_cell_output_dir(self.output_dir, "traces", lat, lon)

        # TODO: isolate loading trace function
        print("Search for trace in\n", outdir_for_cell)
        self.trace = pm.load_trace(outdir_for_cell, model=self.model)

        try:
            for var in ["slope", "intercept", "beta_yearly", "beta_trend", "sigma"]:
                if var not in self.trace.varnames:
                    raise IndexError("Sample data not completely saved. Rerun.")
            print("Successfully loaded sampled data. Skip this for sampling.")
        except IndexError:
            self.sample()
            pm.backends.save_trace(self.trace, outdir_for_cell, overwrite=True)

        self.estimate_timeseries()

        return self.df

    def sample(self):

        TIME0 = datetime.now()

        with self.model:
            self.trace = pm.sample(
                draws=self.draws,
                init=self.init,
                cores=self.cores,
                chains=self.chains,
                tune=self.tune,
                progressbar=self.progressbar,
            )

        TIME1 = datetime.now()
        print(
            "Finished job {0} in {1:.0f} seconds.".format(
                os.getpid(), (TIME1 - TIME0).total_seconds()
            )
        )

        return self.trace

    # def add_season_model(self, modes, smu, sps, beta_name):
    #     """

    #     """
    #     with self.model:
    #         beta = pm.Normal(beta_name, mu=smu, sd=sps, shape=2 * modes)

    def estimate_timeseries(self):

        """ this is a memory-saving version of estimate timeseries.
        caculations are done several times as a trade off for having less
        memory consumtions. """

        # to stay within memory bounds: only take last 1000 samples
        regressor = self.df["gmt_scaled"].values
        subtrace = self.trace[-1000:]
        x_fourier = rescale_fourier(self.df, self.modes)

        self.df["trend"] = rescale(
            (subtrace["slope"] * regressor[:, None]).mean(axis=1), self.df["y"]
        )

        # our posteriors, they do not contain short term variability
        self.df["estimated_scaled"] = (
            subtrace["intercept"]
            + regressor[:, None]
            * (
                subtrace["slope"]
                + det_seasonality_posterior(subtrace["beta_trend"], x_fourier)
            )
            + det_seasonality_posterior(subtrace["beta_yearly"], x_fourier)
        ).mean(axis=1)
        self.df["estimated"] = y_inv(self.df["estimated_scaled"], self.df["y"])

        gmt_driven_trend = (
            regressor[:, None]
            * (
                subtrace["slope"]
                + det_seasonality_posterior(subtrace["beta_trend"], x_fourier)
            )
        ).mean(axis=1)

        # the counterfactual timeseries, our main result
        self.df["cfact_scaled"] = self.df["y_scaled"].data - gmt_driven_trend
        self.df["cfact"] = self.df["y"].data - rescale(gmt_driven_trend, self.df["y"])

        self.df["gmt_driven_trend"] = rescale(gmt_driven_trend, self.df["y"])


def det_dot(a, b):
    """
    The theano dot product and NUTS sampler don't work with large matrices?

    :param a: (np matrix)
    :param b: (theano vector)
    """
    return (a * b[None, :]).sum(axis=-1)


def det_seasonality_posterior(beta, x):
    # FIXME: can this be replaced through det_dot?
    return np.dot(x, beta.T)


def fourier_series(t, p, n):
    # 2 pi n / p
    x = 2 * np.pi * np.arange(1, n + 1) / p
    # 2 pi n / p * t
    x = x * t[:, None]
    x = np.concatenate((np.cos(x), np.sin(x)), axis=1)
    return x

def rescale_fourier(df, modes):

    # TODO: understand what this function does and rename.

    # rescale the period, as t is also scaled
    p = 365.25 / (df["ds"].max() - df["ds"].min()).days
    x = fourier_series(df["t"], p, modes)
    return x

def det_trend(k, m, delta, t, s, A):
    return (k + np.dot(A, delta)) * t + (m + np.dot(A, (-s * delta)))
