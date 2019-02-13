import os
from datetime import datetime
# import xarray as xr
from dtr.dtr_smooth import dtr_smooth
from dtr.dtr_merge import dtr_merge
os.chdir('/home/bschmidt/temp/')

# Get jobs starting time
STIME = datetime.now()
with open('out.txt', 'w') as out:
    out.write('Job started at: ' + str(STIME) + '\n')
print('Job started at: ' + str(STIME))

output = 'gswp3/gmt_gswp3_1901_2010.nc4'
pattern = 'gswp3/tas_gswp3_*'
hws = 182

data = dtr_merge(pattern)
print('\n-----------------------------\n')
print('Data merged from ' +
      str(data['time'][0]) +
      ' to ' +
      str(data['time'][-1]))
print('Data looks like this: ')
print(data)
print('\n-----------------------------\n')
gmt = data['tas'].mean(dim=('lat', 'lon'))
print('-----------------------------')
print('GMT calculated!')
print('Data looks like this: ')
print(gmt)
print('-----------------------------')
gmt_smooth = dtr_smooth(gmt, hws)
print('\n-----------------------------\n')
print('GMT smoothed with half window size: ' + str(hws))
print('Data looks like this: ')
print(gmt_smooth)
print('\n-----------------------------\n')
print('Creating and writing dataset')
gmt_smooth = gmt_smooth.rename({'tas':'gmt'})
gmt_smooth.gmt.attrs['long_name'] = 'moving average of global mean temperature'
gmt_smooth.gmt.attrs['hws'] = str(hws)
print(gmt_smooth)
gmt_smooth.to_netcdf(output, mode='w')
print('Data written to ' + output)
print('Finished!')

# Get jobs finishing time
FTIME = datetime.now()
with open('out.txt', 'a') as out:
    out.write('Job finished at: ' + str(FTIME) + '\n')
print('Job finished at: ' + str(FTIME))
duration = FTIME - STIME
print('Time elapsed ' +
      str(divmod(duration.total_seconds(), 3600)[0]) +
      ' hours!')
