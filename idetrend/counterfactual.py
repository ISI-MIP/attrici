import os

base_compiledir = os.path.expandvars("$HOME/.theano/slot-%d" % (os.getpid() % 500))
os.environ["THEANO_FLAGS"] = "base_compiledir=%s" % base_compiledir

import theano
import numpy as np
import pymc3 as pm
import pandas as pd
import netCDF4 as nc
import idetrend.bayes_detrending as bt
import idetrend.utility as u
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path
import settings as s

class cfact(object):
    def __init__(self, nct, gmt):
        gmt_tdf = create_gmt_frame(nct, gmt)
        self.bayes = bt.bayes_regression(gmt_tdf["gmt_scaled"], s)
        self.gmt = gmt
        self.nct = nct

    def get_data(self, data, i, j):
        self.trace_path = "/p/tmp/mengel/isimip/isi-cfact/output/isicfat_gswp3_tas_full/traces/trace_" + str(i) + "_" + str(j)
        #  self.trace_path = os.path.join(s.output_dir, "traces", "trace_" + str(i) + "_" + str(j))
        self.tdf = bt.create_dataframe(self.nct, data, self.gmt)
        self.model, self.x_data = self.bayes.setup_model(self.tdf)
        print(self.trace_path)

    def load_trace(self):
        with self.model:
            self.trace = pm.load_trace(self.trace_path)

    def det_post(self):
        x_yearly, x_trend = self.x_data
        self.trend_post = self.trace['intercept'] + self.trace['slope'] * self.tdf["gmt_scaled"][:,None]
        self.year_post= det_seasonality_posterior(self.trace['beta_yearly'], x_yearly)
        self.year_trend_post= det_seasonality_posterior(self.trace['beta_trend'], x_trend)

        self.trend_post = u.y_inv(self.trend_post,self.tdf['y'])
        self.year_post = u.y_inv(self.year_post,self.tdf['y']) - self.tdf['y'].min()
        self.year_trend_post = u.y_inv(self.year_trend_post,self.tdf['y']) - self.tdf['y'].min()

        self.post = self.trend_post + self.year_post + self.year_trend_post

    def det_cfact(self):
        #  self.cfact = self.trace['k'] + self.post
        self.cfact = self.tdf["y"].data - self.trend_post.mean(1) - self.year_trend_post.mean(1) + self.trend_post.mean(1)[0]

    def run(self, datazip):
        data, i, j = datazip
        cfact_path_ts = os.path.join(s.output_dir,
                                     "timeseries",
                                     s.cfact_file.split(".")[0]
                                     + "_" + str(i) + "_" + str(j) + ".nc4")

        if not os.path.exists(cfact_path_ts):
            os.makedirs(cfact_path_ts)

        try:
            self.get_data(data, i, j)
            self.load_trace()
            self.det_post()
            post = u.y_inv(self.post, self.tdf['y'])
            self.det_cfact()
            self.save_ts(data, i, j, cfact_path_ts)
            self.plot_cfact_ts(3650, i, j)
            return self.cfact

        except:
            empty = np.empty((data.shape[0],))
            empty[:] = np.nan
            print("trace missing! Printing nans!")
            self.save_ts(empty, i, j, cfact_path_ts)
            return empty

    def plot_cfact_ts(self, last, i, j):
        import matplotlib.dates as mdates

        fig = plt.figure(figsize=(16,10))
        plt.rcParams["font.size"] = 30
        b = 111
        date = self.tdf['ds'].dt.to_pydatetime()

        ax = plt.subplot(b)

        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width, box.height*0.9])
        # plt.title('total, only last 5 years shown')
        plt.plot(date[-last:], self.tdf['y'][-last:], color = "grey", label = "Observated weather")

        p0, = plt.plot(date[-last:], self.post.mean(1)[-last:], lw=4,
                 label="Estimated best guess", color="brown")

        plt.plot(date[-last:], self.cfact[-last:], label="Counterfactual weather")

        plt.legend(loc="upper left", bbox_to_anchor=(0., 1.3),frameon=False)
        plt.ylabel("Regional climatic variable")
        plt.xlabel("Time")

        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        # ax.xaxis.set_ticklabels(ax.get_xticklabels()[::2])
        xt = ax.set_xticks(ax.get_xticks()[::2])
        ax.yaxis.set_ticklabels([])

        # fig.autofmt_xdate()
        plt.ylim(bottom=self.cfact[-last:].min(),
                 top=self.cfact[-last:].max()+4)
        myFmt = mdates.DateFormatter('%m-%Y')
        ax.xaxis.set_major_formatter(myFmt)
        # plt.tight_layout()
        cfact_path_fig = os.path.join(s.output_dir,
                                     "timeseries",
                                     s.cfact_file.split(".")[0]
                                     + "_" + str(i) + "_" + str(j) + ".png")
        plt.savefig(cfact_path_fig,dpi=200)

    def save_ts(self, data, i, j, path):
        cfact_file_ts = nc.Dataset(path, "w", format="NETCDF4")
        tm = cfact_file_ts.createDimension("time", None)
        lat = cfact_file_ts.createDimension("lat", 1)
        lon = cfact_file_ts.createDimension("lon", 1)

        times = cfact_file_ts.createVariable("time", "f8", ("time",))
        longitudes = cfact_file_ts.createVariable("lon", "f8", ("lon",))
        latitudes = cfact_file_ts.createVariable("lat", "f8", ("lat",))
        data_ts = cfact_file_ts.createVariable(s.variable, "f4", ("time", "lat", "lon"))

        latitudes.units = "degree_north"
        latitudes.long_name = "latitude"
        longitudes.standard_name = "latitude"
        longitudes.units = "degree_east"
        longitudes.long_name = "longitude"
        longitudes.standard_name = "longitude"

        latitudes[:] = i/2 + .25
        longitudes[:] = j/2 + .25
        times[:] = range(data.shape[0])
        data_ts[:] = data
        cfact_file_ts.close()



def det_seasonality_posterior(beta, x):
    return np.dot(x, beta.T)


def cfact_helper(data_to_detrend, nct, gmt, i, j):
    data = data_to_detrend.variables[s.variable][:, i, j]
    #  tdf = bt.create_dataframe(nct, data, gmt)
    return (data, i, j)
def create_gmt_frame(nct, gmt):
    ds = pd.to_datetime(
        nct[:], unit="D", origin=pd.Timestamp(nct.units.lstrip("days since"))
    )
    t_scaled = (ds - ds.min()) / (ds.max() - ds.min())
    gmt_on_data_cal = np.interp(t_scaled, np.linspace(0, 1, len(gmt)), gmt)
    gmt_scaled = bt.y_norm(gmt_on_data_cal, gmt_on_data_cal)
    gmt_tdf = pd.DataFrame({"gmt_scaled": gmt_scaled})
    return gmt_tdf
