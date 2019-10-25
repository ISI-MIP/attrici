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
        self.scale_variability = cfg.scale_variability
        self.f_rescale = c.mask_and_scale[cfg.variable][1]
        self.qm_ref_period = cfg.qm_ref_period
        self.save_trace = True
        self.report_mu_sigma = cfg.report_mu_sigma
        self.sigma_model = cfg.sigma_model

        try:
            self.statmodel = model_for_var[self.variable](self.modes, self.sigma_model)
        except KeyError as error:
            print(
                "No statistical model for this variable. Probably treated as part of other variables."
            )
            raise error

    def estimate_parameters(self, df, lat, lon):

        df_valid, gmt_valid = dh.get_valid_subset(df, self.subset)
        self.df_valid = df_valid

        x_fourier = fourier.get_fourier_valid(df, df_valid.index, self.modes)

        self.model = self.statmodel.setup(gmt_valid, x_fourier, df_valid["y_scaled"])

        outdir_for_cell = dh.make_cell_output_dir(
            self.output_dir, "traces", lat, lon, variable=self.variable
        )

        # TODO: isolate loading trace function
        print("Search for trace in\n", outdir_for_cell)
        # As load_trace does not throw an error when no saved data exists, we here
        # test this manually. FIXME: Could be improved, as we check for existence
        # of names and number of chains only, but not that the data is not corrupted.
        try:
            trace = pm.load_trace(outdir_for_cell, model=self.model)
            for var in self.statmodel.vars_to_estimate:
                if var not in trace.varnames:
                    print(var, "is not in trace, rerun sampling.")
                    raise IndexError
            if trace.nchains != self.chains:
                raise IndexError("Sample data not completely saved. Rerun.")
            print("Successfully loaded sampled data. Skip this for sampling.")
        except Exception as e:
            print("Problem with saved trace:", e, ". Redo parameter estimation.")
            trace = self.sample()
            # print(pm.summary(trace)) # takes too much memory
            if self.save_trace:
                pm.backends.save_trace(trace, outdir_for_cell, overwrite=True)

        return trace

    def sample(self):

        TIME0 = datetime.now()

        # with self.model:
        #     trace = pm.sample(
        #         draws=self.draws,
        #         cores=self.cores,
        #         chains=self.chains,
        #         tune=self.tune,
        #         progressbar=self.progressbar,
        #     )

        with self.model:
            mean_field = pm.fit(method='fullrank_advi',
             progressbar=self.progressbar,
            )
            trace = mean_field.sample(1000)

        TIME1 = datetime.now()
        print(
            "Finished job {0} in {1:.0f} seconds.".format(
                os.getpid(), (TIME1 - TIME0).total_seconds()
            )
        )

        return trace

    def estimate_timeseries(self, df, trace, datamin, scale, subtrace=1000):

        trace_for_qm = trace[-subtrace:]

        if trace["mu"].shape[1] < df.shape[0]:
            print("Trace is not complete due to masked data. Resample missing.")
            print(
                "Trace length:", trace["mu"].shape[1], "Dataframe length", df.shape[0]
            )

            xf0 = fourier.rescale(df, self.modes[0])
            xf1 = fourier.rescale(df, self.modes[1])
            xf2 = fourier.rescale(df, self.modes[2])
            xf3 = fourier.rescale(df, self.modes[3])

            with self.model:
                pm.set_data({"xf0": xf0})
                pm.set_data({"xf1": xf1})
                pm.set_data({"xf2": xf2})
                pm.set_data({"xf3": xf3})
                pm.set_data({"gmt": df["gmt_scaled"].values})

                trace_for_qm = pm.sample_posterior_predictive(
                    trace[-subtrace:],
                    samples=subtrace,
                    var_names=["obs", "mu", "sigma"],
                    progressbar=self.progressbar,
                )

        df_mu_sigma = dh.create_ref_df(
            df, trace_for_qm, self.qm_ref_period, self.scale_variability
        )

        cfact_scaled = self.statmodel.quantile_mapping(df_mu_sigma, df["y_scaled"])

        # drops indices that were masked as out of range before
        valid_index = df.dropna().index
        # populate cfact with original values
        df.loc[:, "cfact_scaled"] = df.loc[:, "y_scaled"]
        df.loc[valid_index, "cfact_scaled"] = cfact_scaled[valid_index]

        if (cfact_scaled == np.inf).sum() > 0:
            print(
                "There are",
                (cfact_scaled == np.inf).sum(),
                "values out of range for quantile mapping. Keep original values.",
            )
            df.loc[valid_index[cfact_scaled == np.inf], "cfact_scaled"] = df.loc[
                valid_index[cfact_scaled == np.inf], "y_scaled"
            ]

        # populate cfact with original values
        df.loc[:, "cfact"] = df.loc[:, "y"]
        # overwrite only values adjusted through cfact calculation
        df.loc[valid_index, "cfact"] = self.f_rescale(
            df.loc[valid_index, "cfact_scaled"], datamin, scale
        )

        if self.report_mu_sigma:
            # todo: unifiy indexes so .values can be dropped
            for v in ["mu", "sigma", "mu_ref", "sigma_ref"]:
                df.loc[:, v] = df_mu_sigma.loc[:, v].values

        return df
