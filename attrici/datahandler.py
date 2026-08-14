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


def _calibration_mask(ds, calibration_start=None, calibration_stop=None):
    if calibration_start is None and calibration_stop is None:
        return None
    mask = np.ones(len(ds), dtype=bool)
    if calibration_start is not None:
        mask = mask & (ds >= pd.Timestamp(calibration_start))
    if calibration_stop is not None:
        mask = mask & (ds <= pd.Timestamp(calibration_stop))
    if not mask.any():
        raise ValueError(
            "No data in calibration period [start={}, stop={}]".format(
                calibration_start, calibration_stop
            )
        )
    return mask


def _gmt_on_data_times(ds, gmt, gmt_time=None, calibration_stop=None):
    """Place GMT values on the data timestamps.

    GMT and input already share a calendar range. If lengths match, use GMT as
    is. Otherwise interpolate by date (GMT may be coarser than daily input).

    When ``calibration_stop`` is set, days on or before that date are
    interpolated using only GMT knots up to that date. Otherwise a longer
    application GMT series would pull post-calibration knots (e.g. early 2022)
    into late-2021 days and shift overlap ``gmt_scaled``.
    """
    gmt = np.asarray(gmt, dtype=float).squeeze()
    ds_index = pd.DatetimeIndex(ds)
    ds_asi8 = np.asarray(ds_index.asi8)

    if gmt_time is None:
        if len(gmt) == len(ds_asi8):
            return gmt
        gmt_asi8 = np.linspace(ds_asi8[0], ds_asi8[-1], len(gmt))
        return np.interp(ds_asi8, gmt_asi8, gmt)

    gmt_index = pd.DatetimeIndex(pd.to_datetime(gmt_time))
    gmt_asi8 = np.asarray(gmt_index.asi8)
    if calibration_stop is None:
        return np.interp(ds_asi8, gmt_asi8, gmt)

    cal_stop = pd.Timestamp(calibration_stop)
    cal_knot = gmt_index <= cal_stop
    if not np.any(cal_knot):
        raise ValueError(
            "No GMT samples on or before calibration_stop={}".format(calibration_stop)
        )
    in_cal = ds_index <= cal_stop
    result = np.empty(len(ds_asi8), dtype=float)
    result[in_cal] = np.interp(
        ds_asi8[in_cal], gmt_asi8[cal_knot], gmt[cal_knot]
    )
    after = ~in_cal
    if np.any(after):
        result[after] = np.interp(ds_asi8[after], gmt_asi8, gmt)
    return result


def create_dataframe(
    nct_array, units, data_to_detrend, gmt, variable,
    calibration_start=None, calibration_stop=None, gmt_time=None
):

    # proper dates plus additional time axis that is
    # from 0 to 1 for better sampling performance

    ds = pd.to_datetime(
        nct_array, unit="D", origin=pd.Timestamp(units.lstrip("days since"))
    )

    cal_mask = _calibration_mask(ds, calibration_start, calibration_stop)
    if cal_mask is not None:
        cal_ds = ds[cal_mask]
        cal_span = cal_ds.max() - cal_ds.min()
        if cal_span == pd.Timedelta(0):
            t_scaled = np.zeros(len(ds), dtype=float)
        else:
            t_scaled = (ds - cal_ds.min()) / cal_span
            t_scaled = t_scaled.astype(float)
    else:
        t_scaled = (ds - ds.min()) / (ds.max() - ds.min())

    # GMT already covers the same calendar range as `ds` (validated in
    # run_estimation). Map it onto data times by date, not via t_scaled:
    # t is calibration-anchored, so t=1 is calibration_stop, not the last GMT day.
    # Calibration days ignore post-calibration GMT knots so app_2021 and
    # app_2024 interpolate the overlap the same way.
    gmt_on_data_cal = _gmt_on_data_times(
        ds, gmt, gmt_time, calibration_stop=calibration_stop
    )

    # GMT scaling: use min/max from calibration period only (if set)
    if cal_mask is not None:
        gmt_cal = gmt_on_data_cal[cal_mask]
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

    y_series = pd.Series(data_to_detrend)
    if cal_mask is not None:
        y_scaled, datamin, scale = f_scale(
            y_series, variable, calibration_mask=cal_mask
        )
    else:
        y_scaled, datamin, scale = f_scale(y_series, variable)

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
