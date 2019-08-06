import sys

sys.path.append("..")
from netCDF4 import Dataset
from mpi4py import MPI
import pandas as pd
from pathlib import Path
import settings as s
import numpy as np


def get_ts_list(data_gen):
    """ this function extracts lat and lon values from the names the argument (iterable). It then calculates indices from of respective lon/lat"""
    data_list = []
    lat_indices = []
    lon_indices = []
    for i in data_gen:
        data_list.append(str(i))
        lat_float = float(str(i).split("lat")[-1].split("_")[0])
        lon_float = float(str(i).split("lon")[-1].split(".csv")[0])
        lat_indices.append(int((180 - 2 * lat_float - 0.5) / s.lateral_sub))
        lon_indices.append(int((2 * lon_float - 0.5 + 360) / s.lateral_sub))
    return data_list, lat_indices, lon_indices


# get info about mpi world and process(rank) number
comm = MPI.COMM_WORLD
r = comm.Get_rank()
size = comm.Get_size()

# create output file
out = Dataset(
    Path(s.output_dir) / "cfact" / s.cfact_file,
    "w",
    format="NETCDF4",
    parallel=True,
    set_collective=False,
)

# get input and metadata of timeseries files
path = Path(s.output_dir / "timeseries").glob("**/*.csv")
data_list, lati, loni = get_ts_list(path)
batchsize = len(data_list) / size
headers = pd.read_csv(data_list[0], index_col=0, nrows=1).keys()
headers = headers.drop(["y", "y_scaled", "t", "ds", "gmt", "gmt_scaled"]).values

# get metadata from original data file
#  for r in range(size):
obs = Dataset(Path(s.input_dir) / s.source_file, "r")
time = obs.variables["time"][:]
torigin = obs.variables["time"].units
lat = obs.variables["lat"][:]
lon = obs.variables["lon"][:]
obs.close()
# create/copy metadata to out file (this is automatically performed in collective mode)
out.createDimension("time", time.shape[0])
out.createDimension("lat", lat.shape[0])
out.createDimension("lon", lon.shape[0])

times = out.createVariable("time", "f8", ("time",))
longitudes = out.createVariable("lon", "f8", ("lon",))
latitudes = out.createVariable("lat", "f8", ("lat",))
var_list = []
for var in headers:
    out.createVariable(
        var,
        "f4",
        ("time", "lat", "lon"),
        chunksizes=(time.shape[0], 1, 1),
        fill_value=np.nan,
    )

times.units = torigin
latitudes.units = "degree_north"
latitudes.long_name = "latitude"
latitudes.standard_name = "latitude"
longitudes.units = "degree_east"
longitudes.long_name = "longitude"
longitudes.standard_name = "longitude"

print("This is process", r, "reporting for duty")

#  calculate start and end indices for every batch
if size != 1:
    rest = len(data_list) % (size - 1)
    batchsize = (len(data_list) - rest) / (size - 1)
    print(batchsize)
else:
    batchsize = 0
    rest = len(data_list)
start = int(r * batchsize)
if r != (size - 1):
    end = int((r + 1) * batchsize)
elif r == (size - 1):
    end = int((r * batchsize) + rest)
print("start: ", start)
print("end: ", end)
for i in range(start, end):
    ts = pd.read_csv(data_list[i], index_col=0, engine="c")
    print("lat index: ", lati[i])
    print("lon index: ", loni[i])
    for head in headers:
        out.variables[head][:, lati[i], loni[i]] = ts[head].values

print(r, "wrote data")

out.close()
