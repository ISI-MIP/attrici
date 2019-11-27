import os
import numpy as np
import netCDF4 as nc
from datetime import datetime
from pathlib import Path
import pandas as pd
from func_timeout import func_timeout, FunctionTimedOut
import icounter
import icounter.estimator as est
import icounter.datahandler as dh
import settings as s

print("Version", icounter.__version__)

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

input_file = s.input_dir / s.dataset / s.source_file.lower()
landsea_mask_file = s.input_dir / s.landsea_file

obs_data = nc.Dataset(input_file, "r")
nc_lsmask = nc.Dataset(landsea_mask_file,"r")
nct = obs_data.variables["time"]
lats = obs_data.variables["lat"][:]
lons = obs_data.variables["lon"][:]
longrid, latgrid = np.meshgrid(lons, lats)
jgrid, igrid = np.meshgrid(np.arange(len(lons)), np.arange(len(lats)))

ls_mask = nc_lsmask.variables["LSM"][0,:]
df_specs = pd.DataFrame()
df_specs["lat"] = latgrid[ls_mask==1]
df_specs["lon"] = longrid[ls_mask==1]
df_specs["index_lat"] = igrid[ls_mask==1]
df_specs["index_lon"] = jgrid[ls_mask==1]

print("A total of", len(df_specs), "grid cells to estimate.")

calls_per_arrayjob = np.ones(njobarray) * len(df_specs) // njobarray
if len(df_specs) % njobarray != 0:
    calls_per_arrayjob[-1] = len(df_specs) % njobarray


# Calculate the starting and ending values for this task based
# on the SLURM task and the number of runs per task.
start_num = int(task_id * calls_per_arrayjob[task_id-1])
end_num = int((task_id + 1) * calls_per_arrayjob[task_id-1] - 1)
run_numbers = np.arange(start_num, end_num + 1, 1, dtype=np.int)
print("This is SLURM task", task_id, "which will do runs", start_num, "to", end_num)

estimator = est.estimator(s)

TIME0 = datetime.now()

for n in run_numbers[:]:
    sp = df_specs.loc[n,:]

    # if lat >20: continue
    print("This is SLURM task", task_id, "run number", n, "lat,lon", sp["lat"], sp["lon"])

    data = obs_data.variables[s.variable][:, sp["index_lat"], sp["index_lon"]]
    df, datamin, scale = dh.create_dataframe(nct[:], nct.units, data, gmt, s.variable)

    try:
        trace = func_timeout(
            s.timeout, estimator.estimate_parameters, args=(df, sp["lat"], sp["lon"])
        )
    except (FunctionTimedOut, ValueError) as error:
        print("Sampling at", lat, lon, " timed out or failed.")
        print(error)
        continue

    df_with_cfact = estimator.estimate_timeseries(df, trace, datamin, scale)
    dh.save_to_disk(df_with_cfact, s, sp["lat"], sp["lon"], dformat=s.storage_format)

obs_data.close()
nc_lsmask.close()
print(
    "Estimation completed for all cells. It took {0:.1f} minutes.".format(
        (datetime.now() - TIME0).total_seconds() / 60
    )
)
