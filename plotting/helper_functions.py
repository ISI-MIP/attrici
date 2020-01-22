from pathlib import Path
import argparse
import numpy as np


def get_path(data_dir, var, dataset, runid, tag):
    file_dir = data_dir / runid / "cfact" / var
    if tag == "":
        return file_dir / f"{var}_{dataset.upper()}_cfactual_monmean.nc4"

    else:
        return file_dir / f"{var}_{dataset.upper()}_cfactual_monmean_{tag}.nc4"


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runid", nargs="*", help="provide name of the experiment.")
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="tag in [valid, valid_prthresh] discribes how monthly means are computed",
    )
    parser.add_argument(
        "--rel", action="store_true", help="if specified, plot relative trends"
    )
    parser.add_argument(
        "--cfact", action="store_true", help="if specified, plot cfacts"
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="specify the max value of the plotted scale",
    )
    parser.add_argument(
        "--lat", type=float, help="latitude for which to plot the information"
    )
    parser.add_argument(
        "--lon", type=float, help="longitude for which to plot the information"
    )
    parser.add_argument(
        "--window", type=str, default="1a", help="from ['1a', '1m', '1d']"
    )
    parser.add_argument(
        "--start", type=str, default="01-01-1901", help="date in form dd-mm-yyyy"
    )
    parser.add_argument(
        "--end", type=str, default="31-12-2011", help="date in form dd-mm-yyyy"
    )
    return parser


def get_ylims(ylim):
    return np.array(ylim)[:, 0].min(), np.array(ylim)[:, 1].max()


def get_seasonal_dataset(ds, season):
    ds_season = ds.where(ds["time.season"] == season)
    # rolling mean -> only Jan is not nan
    # however, we loose Jan/ Feb in the first year and Dec in the last
    ds_seas = ds_season.rolling(min_periods=3, center=True, time=3).mean()
    # make annual mean
    return ds_season.groupby("time.year").mean("time")
