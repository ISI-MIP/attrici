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
    task_id = os.environ['SLURM_ARRAY_TASK_ID']
    njobarray = os.environ['SLURM_ARRAY_TASK_COUNT']
except KeyError:
    submitted = False
    # the next two are for testing. remove later.
    njobarray = 1
    task_id = 1

gmt_file = os.path.join(s.input_dir, s.gmt_file)
gmt = bt.get_gmt_on_each_day(gmt_file, s.days_of_year)
gmt_scaled = bt.y_norm(gmt, gmt)

# get data to detrend
to_detrend_file = os.path.join(s.input_dir, s.source_file)
obs_data = nc.Dataset(to_detrend_file, "r")
nct = obs_data.variables["time"]

latsize = obs_data.dimensions["lat"].size
ncells = latsize*obs_data.dimensions["lon"].size

if ncells%njobarray:
    print("task_id",task_id)
    print("njobarray",njobarray)
    print("ncells",ncells)
    raise ValueError("ncells does not fit into array job, adjust jobarray.")

calls_per_arrayjob = ncells/njobarray

# if njobarray != ncells:
#     print("task_id",task_id)
#     print("njobarray",njobarray)
#     print("ncells",ncells)
#     raise ValueError("More jobs than cells were assigned. Check number of jobarray tasks")

# Calculate the starting and ending values for this task based
# on the SLURM task and the number of runs per task.
start_num = (task_id - 1) * (calls_per_arrayjob + 1 )
end_num = task_id*calls_per_arrayjob

# Print the task and run range
print("This is task",task_id,"which will do runs", start_num,"to", end_num)

# Run the loop of runs for this task.
for n in np.arange(start_num,end_num,1, dtype=np.int):
    print("This is SLURM task",task_id,"run number", n)
    i=int(n%latsize)
    j=int(n/latsize)

  #Do your stuff here



# i=task_id%latsize
# j=int(task_id/latsize)
# print(task_id, i, j)
