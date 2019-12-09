import matplotlib.pylab as plt
import cartopy.crs as ccrs
import numpy as np
import netCDF4 as nc
from pathlib import Path
import settings
from plotting.helper_functions import get_path, get_parser

plt.rcParams["figure.figsize"] = 12, 14


def main(runid, tag):
    # data_dir = Path("/p/tmp/mengel/isimip/isi-cfact/output")
    data_dir = settings.output_dir.parents[0]
    variable = settings.variable
    dataset = settings.dataset.lower()
    figure_dir = data_dir / "figures" / runid / "maps"
    figure_dir.mkdir(parents=True, exist_ok=True)
    print(figure_dir)

    ncd = nc.Dataset(get_path(data_dir, variable, dataset, runid, tag), "r")

    # Plotting
    vmax=5e-6
    vmin=None if vmax is None else -vmax
    # define appropriate colorscheme for the given variable
    if variable == 'pr':
        cmap = 'BrBG'
    else:
        cmap = 'coolwarm'

    # y are the original observations, cfact the counterfactual
    fig = plt.figure()
    for i, case in enumerate(["y", "cfact"]):
        data = ncd.variables[case][:]
        # last minus first 30 years
        trend = (data[-30 * 12:, ::-1, ::1].mean(axis=0) -
                 data[0:30 * 12:, ::-1, ::1].mean(axis=0))
        # trend = (np.median(np.array(data[-30 * 12:, ::-1, ::1]), axis=0) -
        #          np.median(np.array(data[0:30 * 12:, ::-1, ::1]), axis=0))

        ax = plt.subplot(211 + i, projection=ccrs.PlateCarree(central_longitude=0.0))
        ax.coastlines()
        img = ax.imshow(trend
                        , vmin=vmin, vmax=vmax
                        , extent=[-180, 180, -90, 90]
                        , cmap=cmap)
        plt.colorbar(img, ax=ax, shrink=0.6)
        ax.grid()
        # ax.plot(loni, lati, "x", markersize=20, markeredgewidth=3, color="r",)
        plt.title(case)

    fig.tight_layout()
    fig.savefig(figure_dir / Path(f"trend_map{tag}.jpg"), dpi=80)


if __name__ == "__main__":
    parser = get_parser()
    o = parser.parse_args()
    if len(o.runid) > 0:
        for runid in o.runid:
            main(runid=runid,tag=o.tag[0])
    else:
        print("no runid provided")
