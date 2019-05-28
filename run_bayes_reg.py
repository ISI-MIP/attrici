import os
import numpy as np
import netCDF4 as nc
from datetime import datetime
import settings as s
import idetrend as idtr
import idetrend.const as c
import idetrend.bayes_detrending as bt
import pymc3 as pm
from mpi4py.futures import MPIPoolExecutor
import sys

# get gmt file
gmt_file = os.path.join(s.data_dir, 'era5_ssa_gmt_leap.nc4')
gmt = bt.get_gmt_on_each_day(gmt_file, s.days_of_year)
gmt_scaled = bt.y_norm(gmt, gmt)
# get data to detrend
to_detrend_file = os.path.join(s.data_dir, s.to_detrend_file)
data = nc.Dataset(to_detrend_file, "r")
# get time data
nct = data.variables["time"]

# combine data to first data table
tdf1 = bt.create_dataframe(nct,
                              data.variables[s.variable][:, 0, :],
                              gmt)
# delete output file if already existent
if os.path.isfile(s.regression_outfile):
    os.remove(s.regression_outfile)
    print("removed old output file")
# output path
file_to_write = os.path.join(s.data_dir, s.regression_outfile)

# set switch for first iteration to allow creation of variables
# in output file based on results
first_iteration = True

if __name__ == "__main__":

    print("Variable is:")
    print(s.variable, flush=True)
    # Create bayesian regression model instance
    bayes = bt.bayes_regression(tdf1["gmt_scaled"])


    # loop through latitudes
    for i in range(data.dimensions["lat"].size):

        print("Latitude index is:")
        print(i, flush=True)
        lat_tdf = bt.create_dataframe(nct,
                                      data.variables[s.variable][:, i, :],
                                      gmt)
        TIME0 = datetime.now()
        if s.mpi:
            with MPIPoolExecutor() as executor:
                futures = executor.map(bayes.mcs,
                                       (bt.mcs_helper(lat_tdf, j)
                                        for j in range(data.dimensions["lon"].size)))

        else:
            print("serial mode")
            futures = map(bayes.mcs, (bt.mcs_helper(lat_tdf, j)
                                        for j in range(data.dimensions["lon"].size)))
                                        #  for j in range(2)))

        futures=list(futures)
        print("finished batch")
        TIME1 = datetime.now()
        duration = TIME1 - TIME0
        print("Sampling models for lat slice " + str(i) + " took", duration.total_seconds(), "seconds.")

        j = 0

        TIME0 = datetime.now()
        for result in futures:
            if first_iteration:
                # create output file
                ds = nc.Dataset(file_to_write, "w", format="NETCDF4")
                coords = (
                    range(s.ndraws * s.nchains),
                    data.variables["lat"],
                    data.variables["lon"]
                )
                bt.create_bayes_reg(ds, futures[0], coords)
                ds.close()
                print("Shaped output file.")
                first_iteration = False

            ds = nc.Dataset(file_to_write, "a")
            var = ds.groups["variables"]
            sampler_stats = ds.groups["sampler_stats"]
            bt.write_bayes_reg(var, sampler_stats, result, (i, j))
            j += 1

        TIME1 = datetime.now()
        duration = TIME1 - TIME0
        print("Saving model parameter traces for lat slice " + str(i) + " took", duration.total_seconds(), "seconds.")
        ds.close()

    data.close()
