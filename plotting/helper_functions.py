from pathlib import Path
import argparse
import numpy as np


def get_path(data_dir, var, dataset, runid):
    return data_dir / Path(runid) / "cfact" / var / Path(
        var + "_" + dataset.upper() + "_cfactual_monmean.nc4")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runid', nargs='*', help='provide name of the experiment.')
    return parser


def get_ylims(ylim):
    return np.array(ylim)[:, 0].min(), np.array(ylim)[:, 1].max()


def get_seasonal_dataset(ds,season):
    ds_season = ds.where(ds['time.season'] == season)
    # rolling mean -> only Jan is not nan
    # however, we loose Jan/ Feb in the first year and Dec in the last
    ds_seas = ds_season.rolling(min_periods=3, center=True, time=3).mean()
    # make annual mean
    return ds_season.groupby('time.year').mean('time')
