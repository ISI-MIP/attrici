import numpy as np
import pymc3 as pm
import matplotlib.pylab as plt
import pandas as pd
import settings as s


class bayes_regression(object):
    def __init__(self, regressor):
        self.regressor = regressor
        self.model = pm.Model()

    def add_linear_model(self, mu=0, sig=5):
        """
        The linear trend implementation in PyMC3.
        :param m: (pm.Model)
        :param regressor: (np.array) MinMax scaled independent variable.
        :return predicted: tt.vector
        """

        with self.model:
            m = pm.Normal('k', mu, sig, testval=0)
            k = pm.Normal('m', mu, sig, testval=0)
            #  return k + m * self.regressor

    def add_sigma(self, beta=.5):
        """
        Adds sigma parameter (HalfCauchy dist) to model.
        :param beta: scale parameter >0
        """
        #  FIXME: Make more flexible by allowing different distributions.

        with self.model:
            sigma = pm.HalfCauchy('sigma', beta, testval=1)
            #  return sigma

    def add_season_model(self, data, modes, beta_name):
        """
        Creates a model of dominant in the data by using
        a fourier series wiht specified number of modes.
        :param data:
        """

        seasonality_prior_scale = 10

        # rescale the period, as t is also scaled
        #  FIXME: Where does the numerator in p come from?
        p = 365.25 / (data['ds'].max() - data['ds'].min()).days
        x = fourier_series(data['t'], p, modes)

        with self.model:
            beta = pm.Normal(beta_name, mu=0,
                             sd=seasonality_prior_scale, shape=2 * modes)
        return x # , beta

    def add_observations(self, data, x_yearly, x_trend):
        with self.model as mod:
            y = (
                mod['k']
                + (mod['m'] * self.regressor)
                + det_dot(x_yearly, mod['beta_yearly'])
                + (data["t"].values * det_dot(x_trend, mod['beta_trend']))
            )
            out = pm.Normal('obs',
                            mu=y,
                            sd=mod['sigma'],
                            observed=data['y_scaled'])
            return out, y

    def find_MAP(self):
        with self.model:
            return pm.find_MAP()

    def mcs(self, data, traces, cores=1, chains=2, progressbar=False,
            live_plot=False):
        # add linear model and sigma
        self.add_linear_model(mu=s.linear_mu)
        print('step 1 done')
        self.add_sigma(beta=s.sigma_beta)
        print('step 2 done')
        # add seasonality models
        x_yearly = self.add_season_model(data, s.modes, beta_name="beta_yearly")
        print('step 3 done')
        x_trend = self.add_season_model(data, s.modes, beta_name="beta_trend")
        print('step 4 done')
        # add observations to finished model
        dist, y = self.add_observations(data, x_yearly, x_trend)
        print('step 5 done')
        with self.model:
            return pm.sample(traces,
                             cores=cores,
                             chains=chains,
                             progressbar=progressbar,
                             live_plot=live_plot)

    #  def write_traces(traces):



def det_dot(a, b):
    """
    The theano dot product and NUTS sampler don't work with large matrices?

    :param a: (np matrix)
    :param b: (theano vector)
    """
    return (a * b[None, :]).sum(axis=-1)


def fourier_series(t, p=s.days_of_year, n=10):
    # 2 pi n / p
    x = 2 * np.pi * np.arange(1, n+1) / p
    # 2 pi n / p * t
    x = x * t[:, None]
    x = np.concatenate((np.cos(x), np.sin(x)), axis=1)
    return x


def sanity_check(m, df):
    """
    :param m: (pm.Model)
    :param df: (pd.DataFrame)
    """
    # Sample from the prior and check of the model is well defined.
    y = pm.sample_prior_predictive(model=m, vars=['obs'])['obs']
    plt.figure(figsize=(16, 6))
    plt.plot(y.mean(0), label='mean prior')
    plt.fill_between(np.arange(y.shape[1]), -y.std(0), y.std(0), alpha=0.25,
                     label='standard deviation')
    plt.plot(df['y_scaled'], label='true value')
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

    ds = pd.to_datetime(nct[:], unit='D',
                        origin=pd.Timestamp(nct.units.lstrip("days since")))
    t_scaled = (ds-ds.min())/(ds.max()-ds.min())
    gmt_on_data_cal = np.interp(t_scaled, np.linspace(0, 1, len(gmt)), gmt)
    gmt_scaled = y_norm(gmt_on_data_cal, gmt_on_data_cal)

    tdf = pd.DataFrame({"ds": ds, "t":t_scaled,
                        "gmt":gmt_on_data_cal,"gmt_scaled":gmt_scaled})

    for i in range(data_to_detrend.shape[1]):
        tdf["y_" + str(i)] = data_to_detrend[:, i]
        y_scaled = y_norm(data_to_detrend[:, i], data_to_detrend[:, i])
        tdf["y_scaled_" + str(i)] = y_scaled

    return tdf


def get_gmt_on_each_day(gmt_file, days_of_year):

    length_of_record = s.endyear - s.startyear + 1

    ncgmt = nc.Dataset(gmt_file, "r")

    # interpolate from yearly to daily values
    gmt_on_each_day = np.interp(
        np.arange(length_of_record * days_of_year),
        ncgmt.variables["time"][:],
        ncgmt.variables["tas"][:],
    )
    ncgmt.close()

    return gmt_on_each_day

def subset_tbf(loni):
    subset = tdf[['ds', 't', 'gmt', 'gmt_scaled', 'y_' + str(i), 'y_scaled_' + str(i)]]
    print('Hallo!', flush=True)
    return subset.rename(columns={'y_' + str(i):'y', 'y_scaled_' + str(i): 'y_scaled'})
