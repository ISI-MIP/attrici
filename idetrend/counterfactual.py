import os

base_compiledir = os.path.expandvars("$HOME/.theano/slot-%d" % (os.getpid() % 500))
os.environ["THEANO_FLAGS"] = "base_compiledir=%s" % base_compiledir

import theano
import numpy as np
import pymc3 as pm
import pandas as pd
import netCDF4 as nc
import idetrend.bayes_detrending as bt
from datetime import datetime
from pathlib import Path
import settings as s

class cfact(object):
    def __init__(self, nct, gmt):
        gmt_tdf = creat_gmt_frame(nct, gmt)
        self.bayes = bt.bayes_regression(gmt_tdf["gmt_scaled"], s.output_dir)
        self.gmt = gmt
        self.nct = nct

    def get_data(self, data, i, j):
        self.trace_path = os.path.join(s.output_dir, "traces", "trace_" + str(i) + "_" + str(j))
        self.tdf = bt.create_dataframe(self.nct, data, self.gmt)
        self.model, self.x_data = self.bayes.setup_model(self.tdf)

    def load_trace(self):
        with self.model:
            self.trace = pm.load_trace(self.trace_path)

    def det_post(self):
        x_yearly, x_trend = self.xdata
        self.trend_post = self.trace['k'] + self.trace['m'] * self.tdf["gmt_scaled"][:,None]
        self.year_post= det_seasonality_posterior(trace['beta_yearly'], x_yearly)
        self.year_trend_post= det_seasonality_posterior(trace['beta_trend'], x_trend)

        self.post = self.trend_post + self.year_post + self.year_trend_post

    def det_cfact(self):
        self.cfact = self.trace['k'] + self.post

    def run(self, datazip):
        data, i, j = datazip
        self.get_data(data, i, j)
        self.load_trace()
        self.det_post()
        post = u.y_inv(self.post, self.tdf['y'])
        self.det_cfact()
        cfact = u.y_inv(self.cfact, self.tdf["y"])

        return cfact.median(1)


    def det_seasonality_posterior(beta, x):
        return np.dot(x, beta.T)


def cfact_helper(data_to_detrend, nct, gmt, i, j):
    data = data_to_detrend.variables[s.variable][:, i, j]
    #  tdf = bt.create_dataframe(nct, data, gmt)
    return (data, i, j)
def creat_gmt_frame(nct, gmt):
    ds = pd.to_datetime(
        nct[:], unit="D", origin=pd.Timestamp(nct.units.lstrip("days since"))
    )
    t_scaled = (ds - ds.min()) / (ds.max() - ds.min())
    gmt_on_data_cal = np.interp(t_scaled, np.linspace(0, 1, len(gmt)), gmt)
    gmt_scaled = bt.y_norm(gmt_on_data_cal, gmt_on_data_cal)
    gmt_tdf = pd.DataFrame({"gmt_scaled": gmt_scaled})
    return gmt_tdf
