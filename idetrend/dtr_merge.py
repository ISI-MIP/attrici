##!/home/bschmidt/.programs/anaconda3/envs/detrending/bin/python3
import xarray as xr
import glob

def dtr_merge(pattern):
    """This function merges netCDF4 data files with a specified pattern.\n
    Input:\n
    pattern:\tPath and pattern of files as you would pass to glob.glob.\n
    Output:\n
    XArray dataset of merge data."""

    # List files
    fl = sorted(glob.glob(pattern))

    # Load data
    # Then concatenate to build timeseries
    for file in fl:
        f = xr.open_dataset(file)
        try:
            data
        except NameError:
            data = f
        else:
            data = xr.concat([data, f], dim='time', data_vars='all')

    # Set attributes to output file as read from input file
    data.attrs = f.attrs
    return data
