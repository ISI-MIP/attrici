import os
import xarray as xr
import dask as da
os.chdir('/home/bschmidt/temp/')


#  @da.delayed
#  def roll_mean(data, output, hws=15):
#      """
#      This functioons computes a rolling mean with speciefied half
#      window size (hws) on xarray datasets (can be used with dask)
#      """
#      # create rolling window object
#      r = data.rolling(time=(2*hws+1), min_periods=1, center=True)
#      # perform operation on rolling window object
#      r = r.mean()
#      print(r)
#      # does not overwrite files (even though it should)
#      delayed_obj = r.to_netcdf(output, mode='w')
#      return delayed_obj


dest_path = '/p/tmp/bschmidt/gswp3/wind_rm_gswp3_1901_2010.nc4'
# open multiple data files into one dataset (chunks specifies chunksize)
f = xr.open_mfdataset('/p/tmp/bschmidt/gswp3/wind_*.nc4',
                      chunks={'lat': 45, 'lon': 45})
# create rolling window object
r = f.rolling(time=31, min_periods=1, center=True)
# perform operation on rolling window object
r = r.mean()
print(r)
# does not overwrite files (even though it should)
delayed_obj = r.to_netcdf(dest_path, mode='w', compute=True)

print('Job Done')
# delayed_obj.visualize()
# compute with progress bar (although not a very good one)
# with ProgressBar():
# results = delayed_obj.compute()

