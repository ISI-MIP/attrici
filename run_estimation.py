import os
import numpy as np
import netCDF4 as nc
from datetime import datetime
from pathlib import Path
import git
from func_timeout import func_timeout, FunctionTimedOut
import icounter.estimator as est
import icounter.datahandler as dh
import settings as s

repo = git.Repo(search_parent_directories=True)
print("Time started:",datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print("git branch:", repo.active_branch)
print("git hash:", repo.head.object.hexsha)

try:
    submitted = os.environ["SUBMITTED"] == "1"
    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    njobarray = int(os.environ["SLURM_ARRAY_TASK_COUNT"])
    s.ncores_per_job = 1
    s.progressbar = False
except KeyError:
    submitted = False
    njobarray = 1
    task_id = 0
    s.progressbar = True

dh.create_output_dirs(s.output_dir)

gmt_file = s.input_dir / s.dataset / s.gmt_file
ncg = nc.Dataset(gmt_file, "r")
gmt = np.squeeze(ncg.variables["tas"][:])
ncg.close()

# get data to detrend
input_file = s.input_dir / s.dataset / s.source_file.lower()
obs_data = nc.Dataset(input_file, "r")
nct = obs_data.variables["time"]
lats = obs_data.variables["lat"][:]
lons = obs_data.variables["lon"][:]
ncells = len(lats) * len(lons)

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
run_numbers = np.arange(start_num, end_num + 1, 1, dtype=np.int)
print("This is SLURM task", task_id, "which will do runs", start_num, "to", end_num)

estimator = est.estimator(s)

TIME0 = datetime.now()

for n in run_numbers[:]:
    i = int(n % len(lats))
    j = int(n / len(lats))
    lat, lon = lats[i], lons[j]

    # if lat >20: continue
    print("This is SLURM task", task_id, "run number", n, "lat,lon", lat, lon)

    data = obs_data.variables[s.variable][:, i, j]
    df, datamin, scale = dh.create_dataframe(nct[:], nct.units, data, gmt, s.variable)

    # Skipping nan-only cells here saves A LOT of time
    if df["y"].size == np.sum(df["y"].isna()):
        print("All data NaN, probably ocean, skip.")
    else:
        try:
            trace = func_timeout(
                s.timeout, estimator.estimate_parameters, args=(df, lat, lon)
            )
        except (FunctionTimedOut, ValueError) as error:
            print("Sampling at", lat, lon, " timed out or failed.")
            print(error)
            continue

        df_with_cfact = estimator.estimate_timeseries(df, trace, datamin, scale)
        # print(df.head(10))
        dh.save_to_disk(df_with_cfact, s, lat, lon, dformat=s.storage_format)

obs_data.close()

print(
    "Estimation completed for all cells. It took {0:.1f} minutes.".format(
        (datetime.now() - TIME0).total_seconds() / 60
    )
)
