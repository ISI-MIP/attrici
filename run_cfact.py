import os
import pickle
import numpy as np
import netCDF4 as nc
from datetime import datetime
import settings as s
import idetrend as idtr
import idetrend.const as c
import idetrend.bayes_detrending as bt
import idetrend.counterfactual as cf
import idetrend.utility as u
import pymc3 as pm
from mpi4py.futures import MPIPoolExecutor
import sys
import pandas as pd

try:
    submitted = os.environ["SUBMITTED"] == "1"
except KeyError:
    submitted = False

# get gmt file
gmt_file = os.path.join(s.input_dir, s.gmt_file)
gmt = nc.Dataset(gmt_file, "r")
gmt = np.squeeze(gmt.variables["tas"][:])

# get data to detrend
source_file = os.path.join(s.data_dir, s.source_file)
data = nc.Dataset(source_file, "r")

latrange = range(data.dimensions["lat"].size)
lonrange = range(data.dimensions["lon"].size)

# get time data and determine dates and years
nct = data.variables["time"]

tdf = bt.create_dataframe(nct, data.variables[s.variable][:, 0, 0], gmt)
if not os.path.exists(s.output_dir):
    os.makedirs(s.output_dir)
    os.makedirs(Path(s.output_dir) / "cfact")

if __name__ == "__main__":
    print("Variable is:")
    print(s.variable, flush=True)

    cfact = cf.cfact(nct, gmt)

    if submitted:
        with MPIPoolExecutor() as executor:
            futures = executor.map(
                cfact.run,
                (cf.cfact_helper(data, i, j) for i in latrange for j in lonrange),
            )

    else:
        print("serial mode")
        futures = map(
            cfact.run,
            (cf.cfact_helper(data, i, j) for i in latrange for j in lonrange),
        )

    futures = list(futures)

    #  create output file
    cfact_path = os.path.join(s.output_dir, s.cfact_file)
    cfact_file = nc.Dataset(cfact_path, "w", format="NETCDF4")
    cfact_file.description = "beta version of counterfactual weather"
    u.copy_nc_container(cfact_file, data)

    k = 0
    for i in latrange:
        for j in lonrange:
            cfact_file.variables[s.variable][:, i, j] = futures[k]
            k += 1

cfact_file.close()
#  trend_file.close()
