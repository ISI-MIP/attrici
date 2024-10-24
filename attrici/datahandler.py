"""
Functions for data handling.
"""

import numpy as np
import pandas as pd
from loguru import logger

import attrici.const as c


def make_cell_output_dir(output_dir, sub_dir, lat, lon, variable):
    """
    Parameters
    ----------
    output_dir : pathlib Path
      The output directory
    sub_dir : string
      Directory one level below output
    lat
      Latitude
    lon
      Longitude
    variable: string
      Variable name
    """

    lat_sub_dir = output_dir / sub_dir / variable / ("lat_" + str(lat))
    lat_sub_dir.mkdir(parents=True, exist_ok=True)

    return lat_sub_dir


def get_subset(df, subset, seed, startdate, stopdate):
    orig_len = len(df)
    if subset > 1:
        np.random.seed(seed)
        subselect = np.random.choice(orig_len, np.int(orig_len / subset), replace=False)
        df = df.loc[np.sort(subselect), :].copy()
    if startdate is None:
        startdate = df.ds[df.ds.first_valid_index()].date()
    if stopdate is None:
        stopdate = df.ds[df.ds.last_valid_index()].date()

    df = df[(df.ds >= str(startdate)) & (df.ds <= str(stopdate))].copy()

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    logger.info("{} data points used from originally {} datapoints.", len(df), orig_len)

    return df


def create_dataframe(ds, data_to_detrend, gmt, variable):
    # use proper dates plus additional time axis that is from 0 to 1 for better
    # sampling performance TODO check
    t_scaled = (ds - ds.min()) / (ds.max() - ds.min())
    gmt_on_data_cal = np.interp(t_scaled, np.linspace(0, 1, len(gmt)), gmt)

    f_scale = c.MASK_AND_SCALE["gmt"][0]
    gmt_scaled, _, _ = f_scale(gmt_on_data_cal, "gmt")

    c.check_bounds(data_to_detrend, variable)
    try:
        f_scale = c.MASK_AND_SCALE[variable][0]
    except KeyError as error:
        logger.error(
            "{} is not implement (yet). Please check if part of the ISIMIP set.",
            variable,
        )
        raise error

    y_scaled, datamin, scale = f_scale(pd.Series(data_to_detrend), variable)

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
    if variable == "pr":
        tdf["is_dry_day"] = np.isnan(y_scaled)  # TODO

    return tdf, datamin, scale


def create_ref_df(df, trace_obs, trace_cfact, params):
    df_params = pd.DataFrame(index=df.index)
    df_params.index = df["ds"]

    for p in params:
        df_params.loc[:, p] = trace_obs[p].mean(axis=0)
        df_params.loc[:, f"{p}_ref"] = trace_cfact[p].mean(axis=0)

    return df_params


def get_cell_filename(outdir_for_cell, lat, lon):
    return outdir_for_cell / f"ts_lat{lat}_lon{lon}.h5"


def save_to_disk(df_with_cfact, fname, lat, lon, **metadata):
    store = pd.HDFStore(fname, mode="w")
    df_name = f"lat_{lat}_lon_{lon}"
    store[df_name] = df_with_cfact
    store.get_storer(df_name).attrs.metadata = metadata
    store.close()
    logger.info("Saved timeseries to {}", fname)
