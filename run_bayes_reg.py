import os
import sys
import numpy as np
import pymc3 as pm
import netCDF4 as nc
from datetime import datetime
from pathlib import Path
from mpi4py.futures import MPIPoolExecutor
import settings as s
import idetrend as idtr
import idetrend.const as c
import idetrend.bayes_detrending as bt

try:
    submitted = os.environ["SUBMITTED"] == "1"
except KeyError:
    submitted = False

# get gmt file
gmt_file = os.path.join(s.input_dir, s.gmt_file)
gmt = bt.get_gmt_on_each_day(gmt_file, s.days_of_year)
gmt_scaled = bt.y_norm(gmt, gmt)
# get data to detrend
to_detrend_file = os.path.join(s.input_dir, s.source_file)
data = nc.Dataset(to_detrend_file, "r")
# get time data
nct = data.variables["time"]

# combine data to first data table
tdf = bt.create_dataframe(nct, data.variables[s.variable][:, 0, 0], gmt)

if not os.path.exists(s.output_dir):
    os.makedirs(s.output_dir)
    os.makedirs(Path(s.output_dir) / "traces")

if __name__ == "__main__":

    print("Variable is:")
    print(s.variable, flush=True)
    # Create bayesian regression model instance
    bayes = bt.bayes_regression(tdf["gmt_scaled"], s.output_dir)

    TIME0 = datetime.now()
    if submitted:

        s.ncores_per_job = 1
        with MPIPoolExecutor() as executor:
            futures = executor.map(
                bayes.run,
                (
                    bt.mcs_helper(nct, data, gmt, i, j)
                    for i in range(data.dimensions["lat"].size)
                    for j in range(data.dimensions["lon"].size)
                ),
            )

    else:
        print("serial mode")
        futures = map(
            bayes.run,
            (
                bt.mcs_helper(nct, data, gmt, i, j)
                for i in range(data.dimensions["lat"].size)
                for j in range(data.dimensions["lon"].size)
            ),
        )

    # next line is necessary to trigger serial map() function.
    futures = list(futures)

    print("Estimation completed for all cells. It took {0:.1f} minutes.".format(
                (datetime.now() - TIME0).total_seconds()/60))

    # if os.path.isfile(s.params_file):
    #     os.remove(s.params_file)
    #     print("removed old output file")

    # # create output file
    # file_to_write = os.path.join(s.output_dir, s.params_file)
    # ds = nc.Dataset(file_to_write, "w", format="NETCDF4")
    # coords = (range(s.ndraws * s.nchains), data.variables["lat"], data.variables["lon"])
    # bt.create_bayes_reg(ds, futures[0], coords)
    # print("Shaped output file.")

    # var = ds.groups["variables"]
    # sampler_stats = ds.groups["sampler_stats"]
    # k = 0
    # for i in range(data.dimensions["lat"].size):
    #     for j in range(data.dimensions["lon"].size):
    #         bt.write_bayes_reg(var, sampler_stats, futures[k], (i, j))
    #         k += 1

    # TIME1 = datetime.now()
    # duration = TIME1 - TIME0
    # print(
    #     "Saving model parameter traces for lat slice " + str(i) + " took",
    #     duration.total_seconds(),
    #     "seconds.",
    # )
    # ds.close()

    # data.close()
