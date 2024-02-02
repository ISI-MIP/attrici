import pathlib
import sys

import netCDF4 as nc
import numpy as np
import pandas as pd

import attrici.const as c
import attrici.fourier as fourier


def create_output_dirs(output_dir):

    """ params: output_dir: a pathlib object """

    for d in ["cfact", "traces", "timeseries"]:
        (output_dir / d).mkdir(parents=True, exist_ok=True)


def make_cell_output_dir(output_dir, sub_dir, lat, lon, variable):

    """ params: output_dir: a pathlib object """

    lat_sub_dir = output_dir / sub_dir / variable / ("lat_" + str(lat))
    lat_sub_dir.mkdir(parents=True, exist_ok=True)

    if sub_dir == "traces":
        #
        return lat_sub_dir / ("lon" + str(lon))
    else:
        return lat_sub_dir


def get_subset(df, subset, seed, startdate):
    orig_len = len(df)
    if subset > 1:
        np.random.seed(seed)
        subselect = np.random.choice(orig_len, np.int(orig_len / subset), replace=False)
        df = df.loc[np.sort(subselect), :].copy()

    if not (startdate is None):
        df = df.loc[startdate:].copy()

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    print(len(df), "data points used from originally", orig_len, "datapoints.")

    return df


def create_dataframe(nct_array, units, data_to_detrend, gmt, variable):

    # proper dates plus additional time axis that is
    # from 0 to 1 for better sampling performance

    ds = pd.to_datetime(
        nct_array, unit="D", origin=pd.Timestamp(units.lstrip("days since"))
    )

    t_scaled = (ds - ds.min()) / (ds.max() - ds.min())
    gmt_on_data_cal = np.interp(t_scaled, np.linspace(0, 1, len(gmt)), gmt)

    f_scale = c.mask_and_scale["gmt"][0]
    gmt_scaled, _, _ = f_scale(gmt_on_data_cal, "gmt")

    c.check_bounds(data_to_detrend, variable)
    try:
        f_scale = c.mask_and_scale[variable][0]
    except KeyError as error:
        print(
            "Error:",
            variable,
            "is not implement (yet). Please check if part of the ISIMIP set.",
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
        tdf["is_dry_day"] = np.isnan(y_scaled)

    return tdf, datamin, scale


def create_ref_df(df, trace_obs, trace_cfact, params):

    df_params = pd.DataFrame(index=df.index)
    df_params.index = df["ds"]

    for p in params:
        df_params.loc[:, p] = trace_obs[p].mean(axis=0)
        df_params.loc[:, f'{p}_ref'] = trace_cfact[p].mean(axis=0)

    return df_params


def get_source_timeseries(data_dir, dataset, qualifier, variable, lat, lon):

    input_file = (
        data_dir
        / dataset
        / pathlib.Path(variable + "_" + dataset.lower() + "_" + qualifier + ".nc4")
    )
    obs_data = nc.Dataset(input_file, "r")
    nct = obs_data.variables["time"]
    lats = obs_data.variables["lat"][:]
    lons = obs_data.variables["lon"][:]
    i = np.where(lats == lat)[0][0]
    j = np.where(lons == lon)[0][0]
    data = obs_data.variables[variable][:, i, j]
    tm = pd.to_datetime(
        nct[:], unit="D", origin=pd.Timestamp(nct.units.lstrip("days since"))
    )
    df = pd.DataFrame(data, index=tm, columns=[variable])
    df.index.name = "Time"
    obs_data.close()
    return df


def get_cell_filename(outdir_for_cell, lat, lon, settings):

    return outdir_for_cell / (
        "ts_" + settings.dataset + "_lat" + str(lat) + "_lon" + str(lon) + settings.storage_format
    )


def test_if_data_valid_exists(fname):

    if ".h5" in str(fname):
        with pd.HDFStore(fname) as store:
            store.get(store.keys()[0])
    elif ".csv" in str(fname):
        pd.read_csv(fname)
    else:
        raise ValueError


def save_to_disk(df_with_cfact, fname, lat, lon, storage_format):

    # outdir_for_cell = make_cell_output_dir(
    #     settings.output_dir, "timeseries", lat, lon, settings.variable
    # )

    # fname = outdir_for_cell / (
    #     "ts_" + settings.dataset + "_lat" + str(lat) + "_lon" + str(lon) + dformat
    # )

    if storage_format == ".csv":
        df_with_cfact.to_csv(fname)
    elif storage_format == ".h5":
        df_with_cfact.to_hdf(fname, "lat_" + str(lat) + "_lon_" + str(lon), mode="w")
    else:
        raise NotImplementedError("choose storage format .h5 or csv.")

    print("Saved timeseries to ", fname)
