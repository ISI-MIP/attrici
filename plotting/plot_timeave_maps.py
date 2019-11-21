import matplotlib.pylab as plt
import cartopy.crs as ccrs
import numpy as np
import netCDF4 as nc
from pathlib import Path
plt.rcParams["figure.figsize"] = 12,14
import os

os.environ['PROJ_LIB'] = '/home/sitreu/anaconda/share/proj'

# data_dir = Path("/p/tmp/mengel/isimip/isi-cfact/output")
data_dir = Path("/home/sitreu/Documents/PIK/Counter Factuals/isi-cfact/output")

figure_dir = data_dir/"figures"/"maps"
figure_dir.mkdir(parents=True, exist_ok=True)
print(figure_dir)

ncd = nc.Dataset(data_dir/"pr_GSWP3_cfactual_monmean.nc4","r")

# Plotting
vmax=1e-5
vmin=-vmax
lati = 8
loni = 4
# y are the original observations, cfact the counterfactual
for i,case in enumerate(["y", "cfact"]):
    data = ncd.variables[case][:]
    # last minus first 30 years
    trend = (data[-30 * 12:, ::-1, ::1].mean(axis=0) -
             data[0:30 * 12:, ::-1, ::1].mean(axis=0))

    ax = plt.subplot(211+i, projection=ccrs.PlateCarree(central_longitude=0.0))
    ax.coastlines()
    img = ax.imshow(trend
                    , vmin=vmin, vmax=vmax
                    , extent=[-180, 180, -90, 90]
                    , cmap='RdBu')
    plt.colorbar(img, ax=ax, shrink=0.6)
    ax.grid()
    ax.plot(loni, lati, "x", markersize=20, markeredgewidth=3, color="r",)
    plt.title(case)

plt.tight_layout()
plt.savefig(figure_dir/"trend_map.jpg",dpi=80)
