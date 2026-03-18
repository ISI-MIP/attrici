import numpy as np
import pandas as pd
import pathlib
import sys
import netCDF4 as nc
import attrici.const as c
import attrici.fourier as fourier


def validate_time_range_alignment(gmt_time, input_time):
    """
    Ensure GMT and input dataset cover the same time range.
    GMT may be at coarser intervals (e.g. every 10th day); tolerance is derived
    from the equidistant GMT interval.

    Parameters
    ----------
    gmt_time : array-like
        GMT timestamps (e.g. pd.DatetimeIndex or numpy datetime64)
    input_time : array-like
        Input dataset timestamps

    Raises
    ------
    ValueError
        If GMT does not fully cover the input time range within tolerance.
    """
    gmt_time = pd.to_datetime(gmt_time)
    input_time = pd.to_datetime(input_time)

    gmt_min = gmt_time.min()
    gmt_max = gmt_time.max()
    input_min = input_time.min()
    input_max = input_time.max()

    if len(gmt_time) < 2:
        raise ValueError(
            "GMT file must have at least 2 time steps to derive interval"
        )

    # Tolerance from equidistant GMT interval
    tolerance = gmt_time[1] - gmt_time[0]

    # GMT start must be within tolerance of input start: input_min - tolerance < gmt_min < input_min + tolerance
    if gmt_min < input_min - tolerance:
        raise ValueError(
            "GMT does not align with input time range: GMT start ({}) is before "
            "input start ({}) by more than GMT interval ({})".format(
                gmt_min, input_min, tolerance
            )
        )
    if gmt_min > input_min + tolerance:
        raise ValueError(
            "GMT does not align with input time range: GMT start ({}) is after "
            "input start ({}) by more than GMT interval ({})".format(
                gmt_min, input_min, tolerance
            )
        )

    # GMT end must be within tolerance of input end: input_max - tolerance < gmt_max < input_max + tolerance
    if gmt_max < input_max - tolerance:
        raise ValueError(
            "GMT does not align with input time range: GMT end ({}) is before "
            "input end ({}) by more than GMT interval ({})".format(
                gmt_max, input_max, tolerance
            )
        )
    if gmt_max > input_max + tolerance:
        raise ValueError(
            "GMT does not align with input time range: GMT end ({}) is after "
            "input end ({}) by more than GMT interval ({})".format(
                gmt_max, input_max, tolerance
            )
        )


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


def get_subset(df, subset, seed, calibration_start, calibration_stop=None):
    orig_len = len(df)
    if subset > 1:
        np.random.seed(seed)
        subselect = np.random.choice(orig_len, np.int(orig_len / subset), replace=False)
        df = df.loc[np.sort(subselect), :].copy()

    if calibration_start is None and calibration_stop is None:
        pass  # no date filtering
    else:
        # Filter by ds column, not index: df has integer index, loc[date:date] would not work
        cal_start = pd.Timestamp(calibration_start) if calibration_start else None
        cal_stop = pd.Timestamp(calibration_stop) if calibration_stop else None
        if cal_start is not None and cal_stop is not None:
            mask = (df["ds"] >= cal_start) & (df["ds"] <= cal_stop)
        elif cal_start is not None:
            mask = df["ds"] >= cal_start
        else:
            mask = df["ds"] <= cal_stop
        df = df.loc[mask].copy()

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    print(len(df), "data points used from originally", orig_len, "datapoints.")

    return df


def create_dataframe(
    nct_array, units, data_to_detrend, gmt, variable,
    calibration_start=None, calibration_stop=None
):

    # proper dates plus additional time axis that is
    # from 0 to 1 for better sampling performance

    ds = pd.to_datetime(
        nct_array, unit="D", origin=pd.Timestamp(units.lstrip("days since"))
    )

    t_scaled = (ds - ds.min()) / (ds.max() - ds.min())
    gmt_on_data_cal = np.interp(t_scaled, np.linspace(0, 1, len(gmt)), gmt)

    # GMT scaling: use min/max from calibration period only (if set)
    if calibration_start is not None or calibration_stop is not None:
        if calibration_start is not None:
            cal_start = pd.Timestamp(calibration_start)
            mask = ds >= cal_start
        else:
            mask = np.ones(len(ds), dtype=bool)
        if calibration_stop is not None:
            cal_stop = pd.Timestamp(calibration_stop)
            mask = mask & (ds <= cal_stop)
        gmt_cal = gmt_on_data_cal[mask]
        if len(gmt_cal) == 0:
            raise ValueError(
                "No GMT data in calibration period [start={}, stop={}]".format(
                    calibration_start, calibration_stop
                )
            )
        gmt_min, gmt_max = gmt_cal.min(), gmt_cal.max()
        scale = gmt_max - gmt_min
        if scale == 0:
            gmt_scaled = np.zeros_like(gmt_on_data_cal)
        else:
            gmt_scaled = (gmt_on_data_cal - gmt_min) / scale
    else:
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
        pd.read_hdf(fname)
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
