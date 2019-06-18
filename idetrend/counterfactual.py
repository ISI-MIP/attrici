import os
import netCDF4 as nc
import numpy as np

import idetrend.bayes_detrending as bt
import idetrend.utility as u
import matplotlib.pyplot as plt
import pandas as pd
import pymc3 as pm
import settings as s

#  import theano


class cfact(object):
    def __init__(self, tdf, cfg):
        """ tdf should contain at least "gmt" and "time vectors" (scaled and unscaled) "y" data should then be passed to the .run() method"""
        self.cfg = cfg
        self.bayes = bt.bayes_regression(tdf["gmt_scaled"], s)
        self.tdf = tdf
        self.gmt = tdf["gmt_scaled"]
        self.trace_path = s.output_dir / "traces"

    def get_data(self, data, i, j):
        #  self.trace_path = os.path.join(s.output_dir, "traces", "trace_" + str(i) + "_" + str(j))
        self.tdf["y_scaled"] = bt.y_norm(data, data)
        self.tdf["y"] = data
        self.model, self.x_data = self.bayes.setup_model(self.tdf)
        print("This is process", os.getpid(), "working on", self.trace_path)

    def load_trace(self, i, j):
        #  with self.bayes.model:
        path = self.trace_path / ("trace_" + str(i) + "_" + str(j))
        self.trace = pm.load_trace(path, model=self.bayes.model)

    def det_post(self):
        x_yearly, x_trend = self.x_data
        trend_post = (
            self.trace["intercept"]
            + self.trace["slope"] * self.tdf["gmt_scaled"][:, None]
        ).mean(1)
        year_post = det_seasonality_posterior(self.trace["beta_yearly"], x_yearly).mean(
            1
        )
        year_trend_post = det_seasonality_posterior(
            self.trace["beta_trend"], x_trend
        ).mean(1)

        trend_post = u.y_inv(trend_post, self.tdf["y"])
        year_post = u.y_inv(year_post, self.tdf["y"]) - self.tdf["y"].min()
        year_trend_post = u.y_inv(year_trend_post, self.tdf["y"]) - self.tdf["y"].min()

        post = trend_post + year_post + year_trend_post
        return trend_post, year_trend_post, post

    def det_cfact(self, trend_post, year_trend_post):
        self.cfact = self.tdf["y"].data - trend_post - year_trend_post + trend_post[0]

    def run(self, datazip):
        data, i, j = datazip
        cfact_path_ts = os.path.join(
            self.cfg.output_dir,
            "timeseries",
            self.cfg.cfact_file.split(".")[0] + "_" + str(i) + "_" + str(j) + ".nc4",
        )
        print(cfact_path_ts)

        #  if not os.path.exists(cfact_path_ts):
        #      os.makedirs(cfact_path)

        #  try:
        self.get_data(data, i, j)
        self.load_trace(i, j)
        trend_post, year_trend_post, post = self.det_post()
        # FIXME: This is already done by y_inv() on different parts in det_post
        #  post = u.y_inv(self.post, self.tdf["y"])
        self.det_cfact(trend_post, year_trend_post)
        self.save_ts(data, i, j, cfact_path_ts)
        #  self.plot_cfact_ts(post, i, j, 40177)
        print("Calc succesfull")
        #  return self.cfact
        return "Done"

        #  except:
        #      empty = np.empty((data.shape[0],))
        #      empty[:] = np.nan
        #      print("trace missing! Printing nans!")
        #      self.save_ts(empty, i, j, cfact_path_ts)
        #      return np.nan

    def save_ts(self, data, i, j, path):
        cfact_file_ts = nc.Dataset(path, "w", format="NETCDF4")
        cfact_file_ts.createDimension("time", None)
        cfact_file_ts.createDimension("lat", 1)
        cfact_file_ts.createDimension("lon", 1)

        times = cfact_file_ts.createVariable("time", "f8", ("time",))
        longitudes = cfact_file_ts.createVariable("lon", "f8", ("lon",))
        latitudes = cfact_file_ts.createVariable("lat", "f8", ("lat",))
        data_ts = cfact_file_ts.createVariable(
            self.cfg.variable,
            "f4",
            ("time", "lat", "lon"),
            chunksizes=(data.shape[0], 1, 1),
        )

        times.units = "days since 1900-01-01 00:00:00"
        latitudes.units = "degree_north"
        latitudes.long_name = "latitude"
        latitudes.standard_name = "latitude"
        longitudes.units = "degree_east"
        longitudes.long_name = "longitude"
        longitudes.standard_name = "longitude"

        # FIXME: make flexible or implement loading from source data
        latitudes[:] = (180 - i) / 2 - 0.25
        longitudes[:] = (j - 360) / 2 + 0.25
        times[:] = range(self.cfact.shape[0])
        data_ts[:] = np.array(self.cfact)
        cfact_file_ts.close()


def det_seasonality_posterior(beta, x):
    return np.dot(x, beta.T)


def cfact_helper(data_to_detrend, i, j):
    data = data_to_detrend.variables[s.variable][:, i, j]
    return (data, i, j)
