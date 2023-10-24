# FIXME: this script has special dependency: pyts
# pyts needs numba and scikit-learn
# they are complicated to handle with the others. Currently solved through
# a second virtual environment. Longer-term, we should use standard packages.

import subprocess
import sys
from datetime import datetime
from pathlib import Path

import netCDF4 as nc
import numpy as np
from pyts.decomposition import SingularSpectrumAnalysis as SSA

# SSA options
# real window size will be window size* (selection step in col)
window_size = 365
grouping = 1
# subset to control smoothness of curve. Subset smaller 4 crashes.
subset = 10

dataset = "GSWP3-W5E5"
output_base = Path("/p/tmp/mengel/isimip/attrici/input/")
output_dir = output_base / dataset

input_file = output_dir / Path("tas_" + dataset.lower() + "_merged.nc4")
mean_file = str(input_file).replace("_merged.nc4", "_gmt.nc4")
ssa_file = str(mean_file).replace("tas_", "").replace("_gmt", "_ssa_gmt")
print(ssa_file)
cmd = "module load cdo && cdo fldmean " + str(input_file) + " " + str(mean_file)
print(cmd)
subprocess.check_call(cmd, shell=True)
print("checked subprocess call")

input_ds = nc.Dataset(mean_file, "r")
col = np.array(np.squeeze(input_ds.variables["tas"][::subset]), ndmin=2)

ssa = SSA(window_size)
print("calculate ssa")
X_ssa = ssa.fit_transform(col)
print("ssa calculated")

output_ds = nc.Dataset(ssa_file, "w", format="NETCDF4")
time = output_ds.createDimension("time", None)
times = output_ds.createVariable("time", "f8", ("time",))
tas = output_ds.createVariable("tas", "f8", ("time"))

output_ds.description = "GMT created from daily values by SSA (10 year step)"
times.units = input_ds.variables["time"].units
times.calendar = input_ds.variables["time"].calendar
times[:] = input_ds.variables["time"][::subset]
tas[:] = X_ssa[0, :]
output_ds.close()
print("Wrote", ssa_file)
