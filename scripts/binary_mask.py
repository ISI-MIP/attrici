# #### AIM: 
# Create binary mask, which is later used to select every third cell from param.nc
# - Binary values:  nan and 1
# - mask.shape: (n*3)+1

import os
import numpy as np
import xarray as xr
import matplotlib.pylab as plt
import subprocess
import settings as s

# define shape of binary mask file
file_len = 16

## generate binary mask of nth * nth cells
x = np.arange(0,file_len)
mask = np.full(len(x)*file_len, np.nan).reshape(file_len, file_len)
mask[::3 , ::3] = np.int64(1)
type(mask[0])
mask


s.output_dir = "/mnt/c/Users/Anna/Documents/UNI/PIK/develop/test_output_correlation"
s.input_dir = "/mnt/c/Users/Anna/Documents/UNI/PIK/develop/test_input/"

## crop existing landseamask file to nth x nth cells

inf = str(s.input_dir) + "/" + s.dataset + f"/landmask_for_testing_{file_len}.nc"
outf = str(s.input_dir) + "/" + s.dataset + f"/b_mask_for_interpolation_{file_len}.nc"

cmd = f"cdo -f nc4c -z zip selindexbox,0,{file_len},0,{file_len} {inf} {outf}"
print(cmd)
subprocess.check_call(cmd, shell=True)

## open cropped landseamask file to overwrite its variable
mask_file = s.input_dir + "/" + s.dataset + f'/landmask_for_testing_{file_len}.nc'
out = s.input_dir + "/" + s.dataset + f'/b_mask_for_interpolation_{file_len}.nc'

mask_file = xr.open_dataset(mask_file)
print(mask_file.variables)#"].shape

# plt.imshow(mask_file.variables["binary_mask"][ :, :])

## overwrite current landseamask variable 
mask_file['area_European_01min'][:] = mask
mask_file.variables["area_European_01min"] #t = t.to_array()

mask_file["binary_mask"] = mask_file["area_European_01min"]
mask_file = mask_file.drop(['area_European_01min'])
mask_file.variables["binary_mask"][:, :]


plt.imshow(mask_file.variables["binary_mask"][ :, :])


mask_file.to_netcdf(out)
mask_file.close()

