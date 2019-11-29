import settings
import warnings
import matplotlib.pyplot as plt
from pathlib import Path
import xarray as xr
from plotting.helper_functions import get_parser, get_path, get_ylims, get_seasonal_dataset

def main(runid, lat, lon):
    # load variables from settings file
    warnings.simplefilter("ignore")
    variable = settings.variable
    dataset = settings.dataset
    data_dir = settings.output_dir

    # load dataset
    ncfl = get_path(data_dir, variable, dataset, runid)
    data = xr.open_dataset(ncfl)
    data = data.sel(lat=lat, lon=lon)
    data["cfact"] *= 1.e6 # Todo @matthias why is that? and for which variable?
    data["y"] *= 1.e6
    data_year = data.groupby('time.year').mean('time')
    # plotting
    figure_dir = data_dir / "figures" / runid
    figure_dir.mkdir(parents=True, exist_ok=True)
    print(figure_dir)
    plt.rcParams["figure.figsize"] = 18, 12
    plt.figure(figsize=(12, 16))
    ylim = []
    axs = []
    ax = plt.subplot(3, 2, 1)
    data_year['y'].plot(label="observed")
    data_year['cfact'].plot(alpha=0.6, label="cfactual")
    ylim.append(ax.get_ylim())
    ax.text(0.05, 0.9, "yearly", transform=ax.transAxes, fontsize=16)
    ax.set_ylabel(variable)
    plt.legend(loc="upper right", frameon=False)
    axs.append(ax)

    for i, season in enumerate(["DJF", "MAM", "JJA", "SON"]):
        data_season = get_seasonal_dataset(data, season)
        ax = plt.subplot(3, 2, i + 3)
        data_season['y'].plot(label="observed")
        data_season['cfact'].plot(alpha=0.6, label="cfactual")
        ylim.append(ax.get_ylim())
        ax.text(0.05, 0.9, season, transform=ax.transAxes, fontsize=16)
        axs.append(ax)
        ax.set_ylabel(variable)

    for ax in axs[1:]:
        ax.set_ylim(get_ylims(ylim))

    plt.savefig(figure_dir / Path(f"{lat};{lon}.jpg"), dpi=80)


if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument('--lat', type=float, help='latitude for which to plot the information')
    parser.add_argument('--lon', type=float, help='longitude for which to plot the information')
    o = parser.parse_args()
    if len(o.runid) > 0:
        for runid in o.runid:
            main(runid=runid, lat=o.lat, lon=o.lon)
    else:
        print('no runid provided')