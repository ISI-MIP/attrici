import os
import numpy as np
import pandas as pd
import pymc3 as pm
from datetime import datetime
import idetrend.datahandler as dh
import idetrend.const as c
import idetrend.models as models
import idetrend.fourier as fourier

model_for_var = {
    "tas": models.Normal,
    "tasrange": models.Normal,
    "tasskew": models.Beta,
    "pr": models.Gamma,
    "prsnratio": models.Beta,
    "hurs": models.Beta,
    "wind": models.Normal,
    "ps": models.Normal,
    "rsds": models.Normal,
    "rlds": models.Normal,
}


class estimator(object):
    def __init__(self, cfg):

        self.output_dir = cfg.output_dir
        self.init = cfg.init
        self.draws = cfg.draws
        self.cores = cfg.ncores_per_job
        self.chains = cfg.chains
        self.tune = cfg.tune
        self.subset = cfg.subset
        self.progressbar = cfg.progressbar
        self.variable = cfg.variable
        # FIXME: make this variable-dependent -> move to models.py
        self.modes = cfg.modes
        self.save_trace = True

        try:
            self.statmodel = model_for_var[self.variable]()
        except KeyError as error:
            print(
                "No statistical model for this variable. Probably treated as part of other variables."
            )
            raise error

    def estimate_parameters(self, df, lat, lon):

        orig_len = len(df)
        df = df.loc[:: self.subset, :].copy()
        # reindex to a [0,1,2, ..] index
        df.reset_index(inplace=True, drop=True)

        x_fourier = fourier.rescale(df, self.modes)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_valid = df.dropna(axis=0, how="any")
        print(
            len(df_valid), "data points used from originally", orig_len, "datapoints."
        )
        regressor = df_valid["gmt_scaled"].values

        self.model = self.statmodel.setup(
            regressor, x_fourier[df_valid.index, :], df_valid["y_scaled"]
        )

        outdir_for_cell = dh.make_cell_output_dir(
            self.output_dir, "traces", lat, lon, variable=self.variable
        )

        # TODO: isolate loading trace function
        print("Search for trace in\n", outdir_for_cell)
        trace = pm.load_trace(outdir_for_cell, model=self.model)

        # As load_trace does not throw an error when no saved data exists, we here
        # test this manually. FIXME: Could be improved, as we check for existence
        # of names and number of chains only, but not that the data is not corrupted.
        try:
            for var in self.statmodel.vars_to_estimate:
                if var not in trace.varnames:
                    raise IndexError("Sample data not completely saved. Rerun.")
            if trace.nchains != self.chains:
                raise IndexError("Sample data not completely saved. Rerun.")
            print("Successfully loaded sampled data. Skip this for sampling.")
        except IndexError:
            trace = self.sample()
            if self.save_trace:
                pm.backends.save_trace(trace, outdir_for_cell, overwrite=True)

        return trace

    def sample(self):

        TIME0 = datetime.now()

        with self.model:
            trace = pm.sample(
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

        return trace

    def estimate_timeseries(self, df, trace, datamin, scale, subtrace=1000):

        regressor = df["gmt_scaled"].values
        x_fourier = fourier.rescale(df, self.modes)

        cfact_scaled = self.statmodel.quantile_mapping(
            trace[subtrace:], regressor, x_fourier, df["y_scaled"]
        )

        return cfact_scaled
