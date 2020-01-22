import matplotlib.pylab as plt
import numpy as np
from pathlib import Path
import xarray as xr
import regionmask as rem
import warnings
import settings
from plotting.helper_functions import (
    get_path,
    get_parser,
    get_ylims,
    get_seasonal_dataset,
)


def main(runid, tag):
    variable = settings.variable
    dataset = settings.dataset
    warnings.simplefilter("ignore")
    plt.rcParams["figure.figsize"] = 18, 12
    # data_dir = Path("/p/tmp/mengel/isimip/isi-cfact/output")
    # data_dir = Path("/home/mengel/data/20190306_IsimipDetrend/output/")
    data_dir = settings.output_dir.parents[0]

    ncfl = get_path(data_dir, variable, dataset, runid, tag)

    ds = xr.open_dataset(ncfl)
    ds["cfact"] *= 1.0e6
    ds["y"] *= 1.0e6

    giorgi_mask = rem.defined_regions.giorgi.mask(ds.cfact)
    giorgi_names = rem.defined_regions.giorgi.names
    land_mask = rem.defined_regions.natural_earth.land_110.mask(ds.cfact) == 0

    figure_dir = data_dir / "figures" / runid
    figure_dir.mkdir(parents=True, exist_ok=True)
    print(figure_dir)

    for rname in giorgi_names[:]:
        print(rname),

        ylim = []
        axs = []
        plt.figure(figsize=(12, 16))
        ds_year = ds.groupby("time.year").mean("time")
        dd = get_season_regmean(ds_year, rname, land_mask, giorgi_mask)
        ax = plt.subplot(3, 2, 1)
        dd[0].plot(label="observed")
        dd[1].plot(alpha=0.6, label="cfactual")
        ylim.append(ax.get_ylim())
        ax.text(0.05, 0.9, rname + " yearly", transform=ax.transAxes, fontsize=16)
        ax.set_ylabel("precipitation")
        plt.legend(loc="upper right", frameon=False)
        axs.append(ax)

        for i, season in enumerate(["DJF", "MAM", "JJA", "SON"]):
            ds_season = get_seasonal_dataset(ds, season)
            dd = get_season_regmean(ds_season, rname, land_mask, giorgi_mask)
            ax = plt.subplot(3, 2, i + 3)
            dd[0].plot()
            dd[1].plot(alpha=0.6)
            ylim.append(ax.get_ylim())
            ax.text(
                0.05, 0.9, rname + " " + season, transform=ax.transAxes, fontsize=16
            )
            axs.append(ax)
            ax.set_ylabel("precipitation")

        for ax in axs[1:]:
            ax.set_ylim(get_ylims(ylim))

        plt.savefig(
            figure_dir / Path(rname.replace(" ", "_") + "_" + tag + ".jpg"), dpi=80
        )


def select_giorgi_by_name(ds, name, land_mask, giorgi_mask):
    """ Select the land pixels belonging to the Giorgi region with name."""
    nregion = rem.defined_regions.giorgi.map_keys(name)
    return ds.where((giorgi_mask == nregion) & land_mask)


def get_season_regmean(ds_season, rname, land_mask, giorgi_mask):
    regy = select_giorgi_by_name(ds_season.y, rname, land_mask, giorgi_mask)
    regcf = select_giorgi_by_name(ds_season.cfact, rname, land_mask, giorgi_mask)
    return regy.mean(dim=("lat", "lon")), regcf.mean(dim=("lat", "lon"))


if __name__ == "__main__":
    parser = get_parser()
    o = parser.parse_args()
    if len(o.runid) > 0:
        for runid in o.runid:
            main(runid=runid, tag=o.tag)
    else:
        print("no runid provided")
