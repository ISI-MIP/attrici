import numpy as np
import pandas as pd
import pathlib
import sys

import icounter.const as c
import icounter.fourier as fourier


def create_output_dirs(output_dir):

    """ params: output_dir: a pathlib object """

    for d in ["cfact", "traces", "timeseries"]:
        (output_dir / d).mkdir(parents=True, exist_ok=True)


def make_cell_output_dir(output_dir, sub_dir, lat, lon, variable=None):

    """ params: output_dir: a pathlib object """

    lat_sub_dir = output_dir / sub_dir / variable / ("lat_" + str(lat))
    lat_sub_dir.mkdir(parents=True, exist_ok=True)

    if sub_dir == "traces":
        #
        return lat_sub_dir / ("lon" + str(lon))
    else:
        return lat_sub_dir


def get_valid_subset(df, modes, subset):

    orig_len = len(df)
    df = df.loc[::subset, :].copy()
    # reindex to a [0,1,2, ..] index
    df.reset_index(inplace=True, drop=True)

    x_fourier = fourier.rescale(df, modes)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_valid = df.dropna(axis=0, how="any")

    print(len(df_valid), "data points used from originally", orig_len, "datapoints.")

    return df_valid, x_fourier[df_valid.index, :], df_valid["gmt_scaled"].values


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

    return tdf, datamin, scale


# def add_cfact_to_df(df, cfact_scaled, datamin, scale, variable):

#     valid_index = df.dropna().index
#     df.loc[valid_index, "cfact_scaled"] = cfact_scaled[valid_index]
#     f_rescale = c.mask_and_scale[variable][1]
#     # populate cfact with original values
#     df["cfact"] = df["y"]
#     # overwrite only values adjusted through cfact calculation
#     df.loc[valid_index, "cfact"] = f_rescale(cfact_scaled[valid_index], datamin, scale)

#     return df


def save_to_disk(df_with_cfact, settings, lat, lon, dformat=".h5"):

    outdir_for_cell = make_cell_output_dir(
        settings.output_dir, "timeseries", lat, lon, settings.variable
    )

    fname = outdir_for_cell / (
        "ts_" + settings.dataset + "_lat" + str(lat) + "_lon" + str(lon) + dformat
    )

    if dformat == ".csv":
        df_with_cfact.to_csv(fname)
    elif dformat == ".h5":
        df_with_cfact.to_hdf(fname, "lat_" + str(lat) + "_lon_" + str(lon), mode="w")
    else:
        raise NotImplementedError("choose storage format .h5 or csv.")

    print("Saved timeseries to ", fname)


def read_from_disk(data_path):

    if data_path.split(".")[-1] == "h5":
        df = pd.read_hdf(data_path)
    elif data_path.split(".")[-1] == "csv":
        df = pd.read_csv(data_path, index_col=0)
    else:
        raise NotImplementedError("choose storage format .h5 or csv.")

    return df


def form_global_nc(ds, time, lat, lon, vnames, torigin):

    ds.createDimension("time", None)
    ds.createDimension("lat", lat.shape[0])
    ds.createDimension("lon", lon.shape[0])

    times = ds.createVariable("time", "f8", ("time",))
    longitudes = ds.createVariable("lon", "f8", ("lon",))
    latitudes = ds.createVariable("lat", "f8", ("lat",))
    for var in vnames:
        data = ds.createVariable(
            var,
            "f4",
            ("time", "lat", "lon"),
            chunksizes=(time.shape[0], 1, 1),
            fill_value=1e20,
        )
    times.units = torigin
    latitudes.units = "degree_north"
    latitudes.long_name = "latitude"
    latitudes.standard_name = "latitude"
    longitudes.units = "degree_east"
    longitudes.long_name = "longitude"
    longitudes.standard_name = "longitude"
    # FIXME: make flexible or implement loading from source data
    latitudes[:] = lat
    longitudes[:] = lon
    times[:] = time
