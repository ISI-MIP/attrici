"""
Estimator class.
"""

import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger

import attrici.const as c
import attrici.datahandler as dh
from attrici import fourier, models
from attrici.modelling import pm

MODEL_FOR_VAR = {
    "tas": models.Tas,
    "tasrange": models.Tasrange,
    "tasskew": models.Tasskew,
    "pr": models.Pr,
    "hurs": models.Hurs,
    "wind": models.Wind,
    "sfcWind": models.Wind,
    "ps": models.Ps,
    "rsds": models.Rsds,
    "rlds": models.Rlds,
}


class Estimator:
    def __init__(
        self,
        output_dir,
        seed,
        progressbar,
        variable,
        modes,
        report_variables,
        start_date,
        stop_date,
    ):
        self.output_dir = output_dir
        self.subset = 1  # TODO ? only use subset = 1 so this can be cleaned up
        self.seed = seed  # TODO ? unused because subset = 1
        self.progressbar = progressbar
        self.variable = variable
        # number of modes in the yearly cycle
        self.modes = modes
        self.report_variables = report_variables
        self.startdate = start_date
        self.stopdate = stop_date

        self.f_rescale = c.MASK_AND_SCALE[self.variable][1]

        try:
            # TODO remove modes from initialization
            self.statmodel = MODEL_FOR_VAR[self.variable](self.modes)

        except KeyError as error:
            logger.error(
                (
                    "No statistical model for this variable. ",
                    "Probably treated as part of other variables.",
                )
            )
            raise error

    def estimate_parameters(self, df, lat, lon, time_0, use_cache=False):
        x_fourier = fourier.get_fourier_valid(df, self.modes)
        x_fourier_01 = (x_fourier + 1) / 2
        x_fourier_01.columns = ["pos" + col for col in x_fourier_01.columns]

        dff = pd.concat([df, x_fourier, x_fourier_01], axis=1)
        df_subset = dh.get_subset(
            dff, self.subset, self.seed, self.startdate, self.stopdate
        )

        self.model = self.statmodel.setup(df_subset)

        if use_cache:
            trace_filename = (
                dh.make_cell_output_dir(
                    self.output_dir, "traces", lat, lon, self.variable
                )
                / f"traces_lat{lat}_lon{lon}.pkl"
            )

        trace = None
        if use_cache and os.path.exists(trace_filename):
            try:
                with open(trace_filename, "rb") as handle:
                    trace = pickle.load(
                        handle
                    )  # TODO use a different format than pickle
            except Exception:
                logger.exception("Problem with saved trace. Redo parameter estimation.")
        if trace is None:
            logger.info(
                "Took {:.1f}s until find_MAP is run",
                (datetime.now() - time_0).total_seconds(),
            )
            trace = pm.find_MAP(model=self.model)
            if use_cache:
                with open(trace_filename, "wb") as handle:
                    free_params = {
                        key: value
                        for key, value in trace.items()
                        if key.startswith("weights") or key == "logp"
                    }
                    pickle.dump(free_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return trace, dff

    def estimate_timeseries(self, df, trace, datamin, scale):
        trace_obs, trace_cfact = self.statmodel.resample_missing(
            trace, df, self.model, self.progressbar
        )

        df_params = dh.create_ref_df(df, trace_obs, trace_cfact, self.statmodel.params)

        cfact_scaled = self.statmodel.quantile_mapping(df_params, df["y_scaled"])
        logger.success("Done with quantile mapping.")

        # fill cfact_scaled as is from quantile mapping
        # for easy checking later
        df.loc[:, "cfact_scaled"] = cfact_scaled

        # rescale all scaled values back to original, invalids included
        df.loc[:, "cfact"] = self.f_rescale(df.loc[:, "cfact_scaled"], datamin, scale)

        # populate invalid values originating from y_scaled with with original values
        if self.variable == "pr":
            df.loc[df["cfact_scaled"] == 0, "cfact"] = 0
        else:
            invalid_index = df.index[df["y_scaled"].isna()]
            df.loc[invalid_index, "cfact"] = df.loc[invalid_index, "y"]

        # df = df.replace([np.inf, -np.inf], np.nan)
        # if df["y"].isna().sum() > 0:
        yna = df["cfact"].isna()
        yinf = df["cfact"] == np.inf
        yminf = df["cfact"] == -np.inf
        logger.info(
            "There are {} NaN values from quantile mapping. Replace.", yna.sum()
        )
        logger.info(
            "There are {} Inf values from quantile mapping. Replace.", yinf.sum()
        )
        logger.info(
            "There are {} -Inf values from quantile mapping. Replace.", yminf.sum()
        )

        df.loc[yna | yinf | yminf, "cfact"] = df.loc[yna | yinf | yminf, "y"]

        # TODO: unifiy indexes so .values can be dropped
        for v in df_params.columns:
            df.loc[:, v] = df_params.loc[:, v].values

        df.loc[:, "logp"] = trace_obs["logp"].mean(axis=0)

        if "all" not in self.report_variables:
            df = df.loc[:, self.report_variables]

        return df
