import settings
import warnings
import matplotlib.pyplot as plt
from pathlib import Path
import xarray as xr
from plotting.helper_functions import get_parser, get_path, get_ylims, get_seasonal_dataset
import cartopy.crs as ccrs

def main(runid, lat, lon):
    # load variables from settings file
    warnings.simplefilter("ignore")
    variable = settings.variable
    dataset = settings.dataset
    data_dir = settings.output_dir

    # load dataset
    ncfl = get_path(data_dir, variable, dataset, runid)
    data = xr.open_dataset(ncfl)
    avail_cell = _get_avail_cell(data, lat, lon)
    print(f"The closest sampled cell to the point lat={lat} lon={lon} "
          f"is: lat={avail_cell['lat']}, lon={avail_cell['lon']}")
    if xr.ufuncs.isnan(data['cfact']).all().item():
        raise ValueError(f'No data for the point: lat={avail_cell["lat"]}, lon={avail_cell["lon"]}')
    data["cfact"] *= 1.e6 # Todo @matthias why is that? and for which variable?
    data["y"] *= 1.e6
    # plotting
    figure_dir = data_dir / "figures" / runid
    figure_dir.mkdir(parents=True, exist_ok=True)
    print(figure_dir)
    plt.rcParams["figure.figsize"] = 18, 12
    plt.figure(figsize=(12, 16))
    ylim = []
    axs = []
    # plot map
    vmax = 10
    vmin = None if vmax is None else -vmax
    for i,case in enumerate(["y", "cfact"]):
        # last minus first 30 years
        trend = (data[case][-30 * 12:, ::-1, ::1].mean(axis=0) -
                 data[case][0:30 * 12:, ::-1, ::1].mean(axis=0))
        # trend = (np.median(np.array(data[-30 * 12:, ::-1, ::1]), axis=0) -
        #          np.median(np.array(data[0:30 * 12:, ::-1, ::1]), axis=0))

        ax = plt.subplot(421+i, projection=ccrs.PlateCarree(central_longitude=0.0))
        ax.coastlines()
        img = ax.imshow(trend
                        , vmin=vmin, vmax=vmax
                        , extent=[-180, 180, -90, 90]
                        , cmap='RdBu')
        plt.colorbar(img, ax=ax, shrink=0.6)
        ax.grid()
        ax.plot(avail_cell['lon'], avail_cell['lat'], "x", markersize=10, markeredgewidth=3, color="r",)
        plt.title(f'{case} map')
        axs.append(ax)

    # plot yearly
    data = data.sel(lat=avail_cell['lat'], lon=avail_cell['lon'])
    data_year = data.groupby('time.year').mean('time')
    ax = plt.subplot(4, 2, 3)
    data_year['y'].plot(label="observed")
    data_year['cfact'].plot(alpha=0.6, label="cfactual")
    ylim.append(ax.get_ylim())
    ax.text(0.05, 0.9, "yearly", transform=ax.transAxes, fontsize=16)
    ax.set_ylabel(variable)
    plt.legend(loc="upper right", frameon=False)
    axs.append(ax)

    for i, season in enumerate(["DJF", "MAM", "JJA", "SON"]):
        data_season = get_seasonal_dataset(data, season)
        ax = plt.subplot(4, 2, i + 5)
        data_season['y'].plot(label="observed")
        data_season['cfact'].plot(alpha=0.6, label="cfactual")
        ylim.append(ax.get_ylim())
        ax.text(0.05, 0.9, season, transform=ax.transAxes, fontsize=16)
        axs.append(ax)
        ax.set_ylabel(variable)

    for ax in axs[2:]:
        ax.set_ylim(get_ylims(ylim))

    plt.savefig(figure_dir / Path(f"{avail_cell['lat']};{avail_cell['lon']}.jpg"), dpi=80)


def _get_avail_cell(data, lat, lon):
    avail_cell = {'lati': abs((data['lat'] - lat)).argmin().item(),
              'loni': abs((data['lon'] - lon)).argmin().item()}
    avail_cell['lat'] = data['lat'][avail_cell['lati']].item()
    avail_cell['lon'] = data['lon'][avail_cell['loni']].item()
    return avail_cell

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