import os
import sys
sys.path.append("..")
from datetime import datetime
from pathlib import Path

import netCDF4 as nc
import numpy as np

import idetrend.bayes_detrending as bt
import idetrend.counterfactual as cf
import settings as s


try:
    submitted = os.environ["SUBMITTED"] == "1"
    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    njobarray = int(os.environ["SLURM_ARRAY_TASK_COUNT"])
except KeyError:
    submitted = False
    njobarray = 1
    task_id = 0
    s.progressbar = True

# get gmt file
gmt_file = os.path.join(s.input_dir, s.gmt_file)
with nc.Dataset(gmt_file, "r") as gmt_obj:
    gmt = np.squeeze(gmt_obj.variables["tas"][:])

# get data to detrend
source_file = os.path.join(s.input_dir, s.source_file)
data = nc.Dataset(source_file, "r")

latsize = data.dimensions["lat"].size
lonsize = data.dimensions["lon"].size
ncells = latsize * data.dimensions["lon"].size

latrange = range(data.dimensions["lat"].size)
lonrange = range(data.dimensions["lon"].size)

# get time data and determine dates and years
nct = data.variables["time"]

tdf = bt.create_dataframe(nct, data.variables[s.variable][:, 0, 0], gmt)

if not os.path.exists(s.output_dir):
    os.makedirs(s.output_dir)
if not os.path.exists(Path(s.output_dir) / "cfact"):
    os.makedirs(Path(s.output_dir) / "cfact")
if not os.path.exists(Path(s.output_dir) / "timeseries"):
    os.makedirs(Path(s.output_dir) / "timeseries")

if ncells % njobarray:
    print("task_id", task_id)
    print("njobarray", njobarray)
    print("ncells", ncells)
    raise ValueError("ncells does not fit into array job, adjust jobarray.")

calls_per_arrayjob = ncells / njobarray

# Calculate the starting and ending values for this task based
# on the SLURM task and the number of runs per task.
start_num = int(task_id * calls_per_arrayjob)
end_num = int((task_id + 1) * calls_per_arrayjob - 1)
print("from", start_num, "to", end_num)

# Print the task and run range
print("This is SLURM task", task_id, "which will do runs", start_num, "to", end_num)

print("Variable is:")
print(s.variable, flush=True)

TIME0 = datetime.now()

cfact = cf.cfact(gmt_tdf, s)

futures = []
for n in np.arange(start_num, end_num + 1, 1, dtype=np.int):
    i = int(n % latsize)
    j = int(n / latsize)
    print("This is SLURM task", task_id, "run number", n, "i,j", i, j)

    data_ij = cf.cfact_helper(data, i, j)
    futr = cfact.run(data_ij)
    futures.append(futr)

print(
    "Estimation completed for all cells. It took {0:.1f} minutes.".format(
        (datetime.now() - TIME0).total_seconds() / 60
    )
)

#  create output file
# FIXME: Either use this path setting to write to different files
#  cfact_path = os.path.join(
#      s.output_dir, s.cfact_file.split(".") + "-" + str(os.getpid()) + ".nc4"
#  )
#  with nc.Dataset(cfact_path, "w", format="NETCDF4") as cfact_file:
#      cfact_file.description = "beta version of counterfactual weather"
#      u.copy_nc_container(cfact_file, data)
#
#      k = 0
#      for i in latrange:
#          for j in lonrange:
#              cfact_file.variables[s.variable][:, i, j] = futures[k]
#          k += 1
#  FIXME: Or this one, that locks the file until the process finished writing
#  cfact_path = os.path.join(s.output_dir, s.cfact_file)
#  with nc.Dataset(cfact_path, "w", format="NETCDF4") as cfact_file:
#      while True:
#          try:
#              fcntl.flock(cfact_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
#              cfact_file.description = "beta version of counterfactual weather"
#              u.copy_nc_container(cfact_file, data)
#
#              k = 0
#              for i in latrange:
#                  for j in lonrange:
#                      cfact_file.variables[s.variable][:, i, j] = futures[k]
#                  k += 1
#              fcntl.flock(cfact_file, fcntl.LOCK_UN)
#              break
#          except IOError as e:
#              if e.errno != errno.EAGAIN:
#                      raise
#              else:
#                  time.sleep(10)
