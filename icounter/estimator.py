import os
import numpy as np
import pandas as pd
import pymc3 as pm
from datetime import datetime

import icounter.datahandler as dh
import icounter.const as c
import icounter.models as models
import icounter.fourier as fourier

model_for_var = {
    "tas": models.Normal,
    "tasrange": models.Rice,
    "tasskew": models.Beta,
    "pr": models.Gamma,
    "prsnratio": models.Beta,
    "hurs": models.Beta,
    "wind": models.Weibull,
    "ps": models.Normal,
    "rsds": models.Normal,
    "rlds": models.Normal,
}


class estimator(object):
    def __init__(self, cfg):

        self.output_dir = cfg.output_dir
        self.draws = cfg.draws
        self.cores = cfg.ncores_per_job
        self.chains = cfg.chains
        self.tune = cfg.tune
        self.subset = cfg.subset
        self.progressbar = cfg.progressbar
        self.variable = cfg.variable
        self.modes = cfg.modes
        self.scale_sigma_with_gmt = cfg.scale_sigma_with_gmt
        self.f_rescale = c.mask_and_scale[cfg.variable][1]

        self.save_trace = True

        try:
            self.statmodel = model_for_var[self.variable](self.modes,
                self.scale_sigma_with_gmt)
        except KeyError as error:
            print(
                "No statistical model for this variable. Probably treated as part of other variables."
            )
            raise error

    def estimate_parameters(self, df, lat, lon):

        df_valid, x_fourier_valid, regressor = dh.get_valid_subset(
            df, self.modes, self.subset
        )

        self.model = self.statmodel.setup(
            regressor, x_fourier_valid, df_valid["y_scaled"]
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
                    print(var, "is not in trace, rerun sampling.")
                    raise IndexError
            if trace.nchains != self.chains:
                raise IndexError("Sample data not completely saved. Rerun.")
            print("Successfully loaded sampled data. Skip this for sampling.")
        except IndexError:
            trace = self.sample()
            print(pm.summary(trace))
            if self.save_trace:
                pm.backends.save_trace(trace, outdir_for_cell, overwrite=True)

        self.df_valid = df_valid
        self.x_fourier_valid = x_fourier_valid
        return trace

    def sample(self):

        TIME0 = datetime.now()

        with self.model:
            trace = pm.sample(
                draws=self.draws,
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

        # regressor = df["gmt_scaled"].values
        # x_fourier = fourier.rescale(df, self.modes)

        cfact_scaled_valid = self.statmodel.quantile_mapping(
            trace[-subtrace:], self.df_valid
        )

        valid_index = self.df_valid.dropna().index
        df.loc[valid_index, "cfact_scaled"] = cfact_scaled_valid

        if (cfact_scaled_valid == np.inf).sum() > 0:
            print("There are", (cfact_scaled == np.inf).sum(),
                  "values out of range for quantile mapping. Keep original values." )
            df.loc[cfact_scaled_valid == np.inf, "cfact_scaled"] = df.loc[cfact_scaled_valid == np.inf,"y_scaled"]

        # populate cfact with original values
        df["cfact"] = df["y"]
        # overwrite only values adjusted through cfact calculation
        df.loc[valid_index, "cfact"] = self.f_rescale(cfact_scaled_valid, datamin, scale)


        return df
