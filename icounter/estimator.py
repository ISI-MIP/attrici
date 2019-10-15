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
        self.qm_ref_period = cfg.qm_ref_period
        self.save_trace = True

        try:
            self.statmodel = model_for_var[self.variable](
                self.modes, self.scale_sigma_with_gmt
            )
        except KeyError as error:
            print(
                "No statistical model for this variable. Probably treated as part of other variables."
            )
            raise error

    def estimate_parameters(self, df, lat, lon):

        df_valid, x_fourier_valid, gmt_valid = dh.get_valid_subset(
            df, self.modes, self.subset
        )

        self.model = self.statmodel.setup(
            gmt_valid, x_fourier_valid, df_valid["y_scaled"]
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
            # print(pm.summary(trace)) # takes too much memory
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

        if self.subset == 1:
            # we can infer parameters directly from the trace
            df_valid = self.df_valid
            trace_for_qm = trace[-subtrace:]
        else:
            # we need to predict through posterior sampling
            print("Posterior-predict deterministic parameters for quantile mapping.")
            # subset fixed to one
            df_valid, x_fourier_valid, gmt_valid = dh.get_valid_subset(
                df, self.modes, 1
            )

            with self.model:
                pm.set_data({"xf": x_fourier_valid})
                pm.set_data({"gmt": gmt_valid})

                trace_for_qm = pm.sample_posterior_predictive(
                    trace[-subtrace:],
                    samples=subtrace,
                    var_names=["obs", "mu", "sigma"],
                )

        cfact_scaled_valid = self.statmodel.quantile_mapping(
            self.qm_ref_period, trace_for_qm, df_valid
        )

        valid_index = df_valid.dropna().index
        # populate cfact with original values
        # df.loc[:,"cfact_scaled"] = df.loc[:,"y_scaled"]
        df.loc[valid_index, "cfact_scaled"] = cfact_scaled_valid

        if (cfact_scaled_valid == np.inf).sum() > 0:
            print(
                "There are",
                (cfact_scaled_valid == np.inf).sum(),
                "values out of range for quantile mapping. Keep original values.",
            )
            df.loc[valid_index[cfact_scaled_valid == np.inf], "cfact_scaled"] = df.loc[
                valid_index[cfact_scaled_valid == np.inf], "y_scaled"
            ]

        # # populate cfact with original values
        df.loc[:, "cfact"] = df.loc[:, "y"]
        # # overwrite only values adjusted through cfact calculation
        df.loc[valid_index, "cfact"] = self.f_rescale(
            df.loc[valid_index, "cfact_scaled"], datamin, scale
        )

        return df
