import shutil
import numpy as np
import pandas as pd
import subprocess
from datetime import datetime
import netCDF4 as nc

def read_from_disk(data_path):

    if data_path.split(".")[-1] == "h5":
        df = pd.read_hdf(data_path)
    elif data_path.split(".")[-1] == "csv":
        df = pd.read_csv(data_path, index_col=0)
    else:
        raise NotImplementedError("choose storage format .h5 or csv.")

    return df



def form_global_nc(ds, time, lat, lon, vnames, torigin):

    # FIXME: can be deleted once merge_cfact is fully replaced by write_netcdf

    ds.createDimension("time", None)
    ds.createDimension("lat", lat.shape[0])
    ds.createDimension("lon", lon.shape[0])

    times = ds.createVariable("time", "f8", ("time",))
    longitudes = ds.createVariable("lon", "f8", ("lon",))
    latitudes = ds.createVariable("lat", "f8", ("lat",))
    for var in vnames:
        data = ds.createVariable(
            var,
            "f4",
            ("time", "lat", "lon"),
            chunksizes=(time.shape[0], 1, 1),
            fill_value=1e20,
        )
    times.units = torigin
    latitudes.units = "degree_north"
    latitudes.long_name = "latitude"
    latitudes.standard_name = "latitude"
    longitudes.units = "degree_east"
    longitudes.long_name = "longitude"
    longitudes.standard_name = "longitude"
    # FIXME: make flexible or implement loading from source data
    latitudes[:] = lat
    longitudes[:] = lon
    times[:] = time

def rechunk_netcdf(ncfile, ncfile_rechunked):


    TIME0 = datetime.now()

    try:
        cmd = (
            "ncks -4 -O --deflate 5 "
            + "--cnk_plc=g3d --cnk_dmn=time,1024 --cnk_dmn=lat,64 --cnk_dmn=lon,128 "
            + str(ncfile)
            + " "
            + ncfile_rechunked
        )
        print(cmd)
        subprocess.check_call(cmd, shell=True)
    except subprocess.CalledProcessError:
        cmd = "module load nco & module load intel/2018.1 && " + cmd
        print(cmd)
        subprocess.check_call(cmd, shell=True)

    print("Rechunking took {0:.1f} minutes.".format((datetime.now() - TIME0).total_seconds() / 60))

    return ncfile_rechunked


def replace_nan_inf_with_orig(variable, source_file, ncfile_rechunked):

    def replace(var, var_orig):

        isinf = np.where(np.isinf(var))
        isnan = np.where(np.isnan(var))

        var[isinf] = var_orig[isinf]
        var[isnan] = var_orig[isnan]

        print(f"Replaced {np.isinf(var).sum()} Inf values." )
        print(f"Replaced {np.isnan(var).sum()} NaN values." )

    ncfile_valid = ncfile_rechunked.rstrip(".nc4") + "_valid.nc4"
    shutil.copy(ncfile_rechunked, ncfile_valid)

    print(f"Replace invalid values in {ncfile_rechunked} with original values from {source_file}")

    ncs = nc.Dataset(source_file, "r")
    ncf = nc.Dataset(ncfile_valid, "a")

    var_orig = ncs.variables[variable]
    var = ncf.variables[variable]

    chunklen = 36
    for xi in range(0,var.shape[1],chunklen):
        for yi in range(0,var.shape[2],chunklen):
            print(xi, yi)
            replace(var[xi:xi+chunklen,yi:yi+chunklen],
                    var_orig[xi:xi+chunklen,yi:yi+chunklen])

    ncs.close()
    ncf.close()
    return ncfile_valid
