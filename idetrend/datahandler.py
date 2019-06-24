import numpy as np
import pandas as pd
import pathlib


def create_output_dirs(output_dir):

    """ params: output_dir: a pathlib object """

    for d in ["traces", "timeseries"]:
        (output_dir / d).mkdir(parents=True, exist_ok=True)


def make_cell_output_dir(output_dir, sub_dir, lat, lon):

    """ params: output_dir: a pathlib object """

    lat_sub_dir = output_dir / sub_dir / ("lat_" + str(lat))
    lat_sub_dir.mkdir(exist_ok=True)

    if sub_dir == "traces":
        #
        return lat_sub_dir / ("cell_lat" + str(lat) + "_lon" + str(lon))
    else:
        return lat_sub_dir

def y_norm(y_to_scale, y_orig):
    return (y_to_scale - y_orig.min()) / (y_orig.max() - y_orig.min())


def y_inv(y, y_orig):
    """rescale data y to y_original"""
    return y * (y_orig.max() - y_orig.min()) + y_orig.min()


def create_dataframe(nct, data_to_detrend, gmt):

    # proper dates plus additional time axis that is
    # from 0 to 1 for better sampling performance

    ds = pd.to_datetime(
        nct[:], unit="D", origin=pd.Timestamp(nct.units.lstrip("days since"))
    )
    t_scaled = (ds - ds.min()) / (ds.max() - ds.min())
    gmt_on_data_cal = np.interp(t_scaled, np.linspace(0, 1, len(gmt)), gmt)
    gmt_scaled = y_norm(gmt_on_data_cal, gmt_on_data_cal)
    y_scaled = y_norm(data_to_detrend, data_to_detrend)

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

    return tdf


def save_to_csv(df_with_cfact, settings, lat, lon):

    outdir_for_cell = make_cell_output_dir(settings.output_dir, "timeseries", lat, lon)

    fname = outdir_for_cell / (
        "ts_"
        + settings.variable
        + "_"
        + settings.dataset
        + "_lat"
        + str(lat)
        + "_lon"
        + str(lon)
        + ".csv"
    )

    df_with_cfact.to_csv(fname)
    print("Timeseries file saved to", fname)
