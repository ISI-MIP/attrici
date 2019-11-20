import matplotlib.pylab as plt
import numpy as np
import netCDF4 as nc
from pathlib import Path

plt.rcParams["figure.figsize"] = 12,14

# data_dir = Path("/p/tmp/mengel/isimip/isi-cfact/output")
data_dir = Path("/home/sitreu/Documents/PIK/Counter Factuals/isi-cfact/output")

figure_dir = data_dir/"figures"/"maps"
figure_dir.mkdir(parents=True, exist_ok=True)
print(figure_dir)

ncd = nc.Dataset(data_dir/"pr_GSWP3_cfactual_monmean.nc4","r")

# Plotting
sb = 1
vmax=4e-5
vmin=-vmax
lati = 8
loni = 4
fig, ax = plt.subplots(nrows=2,ncols=1)
# y are the original observations, cfact the counterfactual
for i,case in enumerate(["y", "cfact"]):
    data = ncd.variables[case][:]
    # last minus first 30 years
    trend = (data[-30 * 12:, ::sb, ::sb].mean(axis=0) -
             data[0:30 * 12:, ::sb, ::sb].mean(axis=0))
    img = ax[i].imshow(trend,vmin=vmin,vmax=vmax)
    plt.colorbar(img, ax=ax[i], shrink=0.6)
    ax[i].grid()
    ax[i].plot(loni, lati, "x", markersize=20, markeredgewidth=3, color="r",)

plt.tight_layout()
plt.savefig(figure_dir/"trend_map.jpg",dpi=80)
