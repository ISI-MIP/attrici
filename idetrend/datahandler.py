import numpy as np
import pandas as pd
import pathlib
import sys

# sys.path.append("..")
import idetrend.const as c


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

# def normalize_to_unity(data):

#     """ Take a pandas Series and scale it linearly to
#     lie within [0, 1]. Return pandas Series as well as the
#     data minimum and the scale. """

#     scale = data.max() - data.min()
#     scaled_data = (data - data.min())/scale

#     return scaled_data, data.min(), scale

# def undo_normalization(scaled_data, datamin, scale):

#     """ Use a given datamin and scale to rescale to original. """

#     return scaled_data*scale + datamin

# def y_norm(y_to_scale, y_orig):
#     return (y_to_scale - y_orig.min()) / (y_orig.max() - y_orig.min())


# def y_inv(y, y_orig):
#     """rescale data y to y_original"""
#     return y * (y_orig.max() - y_orig.min()) + y_orig.min()

def create_dataframe(nct, data_to_detrend, gmt, variable):

    # proper dates plus additional time axis that is
    # from 0 to 1 for better sampling performance

    if nct.__class__.__name__ == "Variable":
        ds = pd.to_datetime(
            nct[:], unit="D", origin=pd.Timestamp(nct.units.lstrip("days since"))
        )
    else:
        ds = nct
    t_scaled = (ds - ds.min()) / (ds.max() - ds.min())
    gmt_on_data_cal = np.interp(t_scaled, np.linspace(0, 1, len(gmt)), gmt)

    f_scale = c.mask_and_scale["gmt"][0]
    gmt_scaled, _, _ = f_scale(gmt_on_data_cal)
    f_scale = c.mask_and_scale[variable][0]
    y_scaled, datamin, scale = f_scale(data_to_detrend)

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

def add_cfact_to_df(df, cfact_scaled, datamin, scale, variable):

    valid_index = df.dropna().index
    df.loc[valid_index, "cfact_scaled"] = cfact_scaled[valid_index]
    f_rescale = c.mask_and_scale[variable][1]
    # populate cfact with original values
    df["cfact"] = df["y"]
    # overwrite only values adjusted through cfact calculation
    df.loc[valid_index, "cfact"] = f_rescale(cfact_scaled[valid_index], datamin, scale)

    return df

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
            fill_value=np.nan,
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
