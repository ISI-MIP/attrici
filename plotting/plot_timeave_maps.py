import matplotlib.pylab as plt
from matplotlib import colors
import pymc3 as pm
import numpy as np
# import sys
import netCDF4 as nc
from pathlib import Path
import arviz as az
import argparse

import icounter.models as models
import icounter.fourier as fourier
import icounter.datahandler as dh
import icounter.const as c
plt.rcParams["figure.figsize"] = 12,14

# data_dir = Path("/p/tmp/mengel/isimip/isi-cfact/output")
data_dir = Path("/home/mengel/data/20190306_IsimipDetrend/output/")

def get_path(data_dir, var, dataset, runid):
    return data_dir/Path(runid)/"cfact"/var/Path(
        var+"_"+dataset.upper()+"_cfactual_monmean.nc4")

variable="pr"
dataset="gswp3"

parser = argparse.ArgumentParser()
parser.add_argument('runid', nargs='*', help='provide name of the experiment.')
o = parser.parse_args()
runid=o.runid[0]
# print(runid)
figure_dir = data_dir/"figures"/runid/"maps"
figure_dir.mkdir(parents=True, exist_ok=True)
print(figure_dir)

ncd = nc.Dataset(get_path(data_dir, variable, dataset, runid),"r")

sb = 1
data = ncd.variables["y"][:]
trend = (data[-30*12:,::sb,::sb].mean(axis=0) -
         data[0:30*12:,::sb,::sb].mean(axis=0))

vmax=1e-5
vmin = None if vmax==None else -vmax
ax1 = plt.subplot(211)
# last minus first 30 years
plt.imshow(trend,vmin=vmin,vmax=vmax)
#            ,norm=colors.SymLogNorm(linthresh=1e-7, linscale=1e-7,vmin=-1e-5, vmax=1e-5))
plt.colorbar(shrink=0.6)
# plt.plot(loni, lati, "x", markersize=20, markeredgewidth=3, color="r",)
plt.grid()
plt.title(runid)

case = "cfact"

data_cf = ncd.variables[case][:]
trend_cfact = (data_cf[-30*12:,::sb,::sb].mean(axis=0)
                    - data_cf[0:30*12:,::sb,::sb].mean(axis=0))

lati = 8
loni = 4
ax2 = plt.subplot(212)

# y are the original observations, cfact the counterfactual
# last minus first 30 years
plt.imshow(trend_cfact,vmin=vmin,vmax=vmax)
plt.colorbar(shrink=0.6)
plt.plot(loni, lati, "x", markersize=20, markeredgewidth=3, color="r",)
plt.grid()
plt.tight_layout()
plt.savefig(figure_dir/"trend_map.jpg",dpi=80)
