print("Beginning of run_bayes_reg.py")
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

#  Data Paths (now ignored for bayes implementation)
#  gmt_file = os.path.join(s.data_dir, s.gmt_file)
#  to_detrend_file = os.path.join(s.data_dir, s.to_detrend_file)
#  gmt_on_each_day = idtr.utility.get_gmt_on_each_day(gmt_file, s.days_of_year)
#  data = nc.Dataset(to_detrend_file, "r")
#  data_to_detrend = data.variables[s.variable]  # [:]
#  data_to_detrend = idtr.utility.check_data(data_to_detrend, to_detrend_file)
gmt_file = os.path.join(s.data_dir, 'era5_ssa_gmt_leap.nc4')
to_detrend_file = os.path.join(s.data_dir, s.to_detrend_file)
data = nc.Dataset(to_detrend_file, "r")
data_to_detrend = data.variables[s.variable][:, 10, :]
nct = data.variables["time"]
ncg = nc.Dataset(gmt_file, "r")
tdf = bt.create_dataframe(nct, data_to_detrend, ncg.variables["tas"][:])
#  print(data_to_detrend.shape)
data.close()
ncg.close()

if __name__ == "__main__":

    TIME0 = datetime.now()
    print("Variable is:")
    print(s.variable, flush=True)

    # Create bayesian regression model instance
    bayes = bt.bayes_regression(tdf["gmt_scaled"])

    #  print(bt.subset_tbf(tdf, 0))
    with MPIPoolExecutor() as executor:
        future = executor.map(bayes.mcs,
                              (bt.mcs_helper(tdf, i, 1000) for i in range(data_to_detrend.shape[1])))
    print(future.result())
    #  results = idtr.utility.run_regression_on_dataset(
    #      data_to_detrend, s.days_of_year, bayes, s.n_jobs
    #  )
    #  traces = bayes.mcs(s.ntraces, s.

    # And run the sanity check
    #  bt.sanity_check(bayes.model, tdf)

    #  model = pm.Model()
    #
    #  with model:
    #      y = bt.trend_model(model, tdf["gmt_scaled"])
    #
    #      sigma = pm.HalfCauchy('sigma', 0.5, testval=1)
    #      pm.Normal('obs',
    #                   mu=y,
    #                   sd=sigma,
    #                   observed=tdf["y_scaled"])
    #  regr = idtr.regression.regression(
    #      gmt_on_each_day,
    #      s.min_ts_len,
    #      c.minval[s.variable],
    #      c.maxval[s.variable],
    #      c.transform[s.variable],
    #  )
    #
    #  results = idtr.utility.run_regression_on_dataset(
    #      data_to_detrend, s.days_of_year, regr, s.n_jobs
    #  )
    #
    #  TIME1 = datetime.now()
    #  duration = TIME1 - TIME0
    #  print("Calculation took", duration.total_seconds(), "seconds.")
    #
    #  file_to_write = os.path.join(s.data_dir, s.regression_outfile)
    #
    #  if os.path.exists(file_to_write):
    #      os.remove(file_to_write)
    #
    #  idtr.regression.write_regression_stats(
    #      data_to_detrend.shape,
    #      (data.variables["lat"], data.variables["lon"]),
    #      results,
    #      file_to_write,
    #      s.days_of_year,
    #  )
    #  TIME2 = datetime.now()
    #  duration = TIME2 - TIME1
    #  print("Saving took", duration.total_seconds(), "seconds.")
