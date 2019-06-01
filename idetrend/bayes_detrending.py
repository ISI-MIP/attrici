import os

base_compiledir = os.path.expandvars("$HOME/.theano/slot-%d" % (os.getpid() % 500))
os.environ["THEANO_FLAGS"] = "base_compiledir=%s" % base_compiledir

import theano
import numpy as np
import pymc3 as pm
import matplotlib.pylab as plt
import pandas as pd
import netCDF4 as nc
from datetime import datetime
from pathlib import Path
import settings as s

#  print(theano.config)


class bayes_regression(object):

    def __init__(self, regressor, output_dir):

        self.regressor = regressor
        self.output_dir = output_dir

    def add_linear_model(self, mu=0, sig=5):
        """
        The linear trend implementation in PyMC3.
        :param m: (pm.Model)
        :param regressor: (np.array) MinMax scaled independent variable.
        :return predicted: tt.vector
        """

        with self.model:

            k = pm.Normal("k", mu, sig)
            m = pm.Normal("m", mu, sig)
            #  return k + m * self.regressor

    def add_sigma(self, beta=0.5):
        """
        Adds sigma parameter (HalfCauchy dist) to model.
        :param beta: scale parameter >0
        """
        #  FIXME: Make more flexible by allowing different distributions.

        with self.model:
            sigma = pm.HalfCauchy("sigma", beta, testval=1)
            #  sigma = pm.HalfStudentT('sigma', lam=4, nu=10)
            #  return sigma

    def add_season_model(self, data, modes, smu, sps, beta_name):
        """
        Creates a model of dominant in the data by using
        a fourier series wiht specified number of modes.
        :param data:
        """

        # rescale the period, as t is also scaled
        p = 365.25 / (data["ds"].max() - data["ds"].min()).days
        x = fourier_series(data["t"], p, modes)

        with self.model:
            beta = pm.Normal(beta_name, mu=smu, sd=sps, shape=2 * modes)
        return x  # , beta

    def add_observations(self, data, x_yearly, *x_trend):
        with self.model as mod:
            y = (
                mod["k"]
                + mod["m"] * self.regressor
                + det_dot(x_yearly, mod["beta_yearly"])
                + (data["t"].values * det_dot(x_trend, mod["beta_trend"]))
            )
            out = pm.Normal("obs", mu=y, sd=mod["sigma"], observed=data["y_scaled"])
            return out, y

    def find_MAP(self):
        with self.model:
            return pm.find_MAP()

    def run(self, datazip):

        data, i, j = datazip
        self.setup_model(data)
        self.sample(i, j)
        self.save_trace(i, j)

    def setup_model(self, data):

        # create instance of pymc model class
        self.model = pm.Model()
        # add linear model and sigma
        self.add_linear_model(mu=s.linear_mu)
        self.add_sigma(beta=s.sigma_beta)
        # add seasonality models
        x_yearly = self.add_season_model(
            data, s.modes, smu=s.smu, sps=s.sps, beta_name="beta_yearly"
        )
        x_trend = self.add_season_model(
            data, s.modes, smu=s.stmu, sps=s.stps, beta_name="beta_trend"
        )
        # add observations to finished model
        dist, y = self.add_observations(data, x_yearly, x_trend)

        return self.model

    def sample(
        self, i, j,
        init=s.init,
        draws=s.ndraws,
        cores=s.ncores_per_job,
        chains=s.nchains,
        tune=s.ntunes,
        progressbar=s.progressbar,
        live_plot=s.progressbar,
    ):

        print("Working on [", i, j, "]", flush=True)
        TIME0 = datetime.now()

        with self.model:
            self.trace = pm.sample(
                draws=draws,
                init=init,
                cores=cores,
                chains=chains,
                tune=tune,
                progressbar=progressbar,
                live_plot=live_plot,
            )
        TIME1 = datetime.now()
        print(
            "Finished job {0} in {1:.0f} seconds.".format(
                os.getpid(), (TIME1 - TIME0).total_seconds()
            )
        )

        return self.trace

    def save_trace(self, i, j):

        output_dir = self.output_dir / "traces" / ("trace_"+str(i)+"_"+str(j))
        pm.backends.save_trace(self.trace, output_dir, overwrite=True)


def det_dot(a, b):
    """
    The theano dot product and NUTS sampler don't work with large matrices?

    :param a: (np matrix)
    :param b: (theano vector)
    """
    return (a * b[None, :]).sum(axis=-1)


def fourier_series(t, p=s.days_of_year, n=10):
    # 2 pi n / p
    x = 2 * np.pi * np.arange(1, n + 1) / p
    # 2 pi n / p * t
    x = x * t[:, None]
    x = np.concatenate((np.cos(x), np.sin(x)), axis=1)
    return x


# TODO: move these helper functions to other files.


def sanity_check(m, df):
    """
    FIXME: plotting should not appear in the main code files.

    :param m: (pm.Model)
    :param df: (pd.DataFrame)
    """
    # Sample from the prior and check of the model is well defined.
    y = pm.sample_prior_predictive(model=m, vars=["obs"])["obs"]
    plt.figure(figsize=(16, 6))
    plt.plot(y.mean(0), label="mean prior")
    plt.fill_between(
        np.arange(y.shape[1]),
        -y.std(0),
        y.std(0),
        alpha=0.25,
        label="standard deviation",
    )
    plt.plot(df["y_scaled"], label="true value")
    plt.legend()


# Determine g, based on the parameters
def det_trend(k, m, delta, t, s, A):
    return (k + np.dot(A, delta)) * t + (m + np.dot(A, (-s * delta)))


def y_norm(y_to_scale, y_orig):
    return (y_to_scale - y_orig.min()) / (y_orig.max() - y_orig.min())


def y_inv(y_to_scale, y_orig):
    return y_to_scale * (y_orig.max() - y_orig.min()) + y_orig.min()


def create_dataframe(nct, data_to_detrend, gmt):

    # proper dates plus additional time axis that is
    # from 0 to 1 for better sampling performance

    ds = pd.to_datetime(
        nct[:], unit="D", origin=pd.Timestamp(nct.units.lstrip("days since"))
    )
    t_scaled = (ds - ds.min()) / (ds.max() - ds.min())
    gmt_on_data_cal = np.interp(t_scaled, np.linspace(0, 1, len(gmt)), gmt)
    gmt_scaled = y_norm(gmt_on_data_cal, gmt_on_data_cal)
    y_scaled = y_norm(data_to_detrend, data_to_detrend)

    tdf = pd.DataFrame(
        {
            "ds": ds,
            "t": t_scaled,
            "y": data_to_detrend,
            "y_scaled": y_scaled,
            "gmt": gmt_on_data_cal,
            "gmt_scaled": gmt_scaled,
        }
    )

    return tdf


# def get_gmt_on_each_day(gmt_file, days_of_year):

#     length_of_record = s.endyear - s.startyear + 1

#     ncgmt = nc.Dataset(gmt_file, "r")

#     # interpolate from yearly to daily values
#     gmt_on_each_day = np.interp(
#         np.arange(length_of_record * days_of_year),
#         ncgmt.variables["time"][:],
#         np.squeeze(ncgmt.variables["tas"][:]),
#     )
#     ncgmt.close()

#     return gmt_on_each_day


def mcs_helper(nct, data_to_detrend, gmt, i, j):

    data = data_to_detrend.variables[s.variable][:, i, j]
    tdf = create_dataframe(nct, data, gmt)
    return (tdf, i, j)


def create_bayes_reg(ds, data, original_data_coords):

    trace = ds.createDimension("trace", None)
    nchain = ds.createDimension("nchain", None)
    lat = ds.createDimension("lat", original_data_coords[1].shape[0])
    lon = ds.createDimension("lon", original_data_coords[2].shape[0])

    s = ds.createGroup("sampler_stats")
    v = ds.createGroup("variables")

    # create variables for group "variables"
    traces = ds.createVariable("traces", "u1", ("trace",))
    nchain = ds.createVariable("nchains", "u1", ("nchain"))
    longitudes = ds.createVariable("lon", "f4", ("lon",))
    latitudes = ds.createVariable("lat", "f4", ("lat",))

    variables = []
    for varname in data.varnames:
        if data.get_values(varname).ndim == 1:
            variables.append(v.createVariable(varname, "f4", ("trace", "lat", "lon")))
        elif data.get_values(varname).ndim == 2:
            variables.append(
                v.createVariable(varname, "f4", ("trace", "nchain", "lat", "lon"))
            )
    stats = []
    for stat in data.stat_names:
        stats.append(s.createVariable(stat, "f4", ("trace", "lat", "lon")))

    ds.description = "bayesian regression test script"
    # ds.history = "Created " + time.ctime(time.time())
    latitudes.units = "degrees north"
    longitudes.units = "degrees east"
    traces.units = "."

    tras = original_data_coords[0][:]
    lats = original_data_coords[1][:]
    lons = original_data_coords[2][:]

    traces[:] = tras
    latitudes[:] = lats
    longitudes[:] = lons


def write_bayes_reg(vargroup, statgroup, trace, indices):
    print("writing to indices:")
    print(indices[0])
    print(indices[1])
    for varname in trace.varnames:
        if trace.get_values(varname).ndim == 1:
            vargroup.variables[varname][:, indices[0], indices[1]] = trace.get_values(
                varname
            )
        elif trace.get_values(varname).ndim == 2:
            vargroup.variables[varname][
                :, :, indices[0], indices[1]
            ] = trace.get_values(varname)
    for stat_name in trace.stat_names:
        statgroup.variables[stat_name][
            :, indices[0], indices[1]
        ] = trace.get_sampler_stats(stat_name)
