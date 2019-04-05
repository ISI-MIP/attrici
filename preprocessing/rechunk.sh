#!/bin/bash

#SBATCH --qos=priority
##SBATCH --partition=priority
#SBATCH --job-name=rechunk
#SBATCH --account=isipedia
#SBATCH --output=../output/rechunk.out
#SBATCH --error=../output/rechunk.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bschmidt@pik-potsdam.de

# # block one node completely to get all its memory.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#
# run from parent directory
if [ -e settings.py ]; then
    settings_file=settings.py 
else 
    settings_file=../settings.py
fi 

# Get information from settings file
# sed gets rid of string markers (',")
variable="$(grep 'variable =' ${settings_file} | cut -d' ' -f3 | sed "s/'//g" | sed 's/"//g')"
datafolder="$(grep 'data_dir =' ${settings_file} | cut -d' ' -f3 | grep $USER | sed "s/'//g" | sed 's/"//g')"
dataset="$(grep 'dataset =' ${settings_file} | cut -d' ' -f3 | sed "s/'//g" | sed 's/"//g')"
startyear="$(grep 'startyear =' ${settings_file} | cut -d' ' -f3 | sed "s/'//g" | sed 's/"//g')"
endyear="$(grep 'endyear =' ${settings_file} | cut -d' ' -f3 | sed "s/'//g" | sed 's/"//g')"

echo 'Rechunk the following variable'
echo $variable
nccopy -u -k 'nc4' -m 32G -c time/4018,lat/1,lon/720 ${datafolder}${variable}_${dataset}_${startyear}_${endyear}_noleap.nc4 \
    ${datafolder}${variable}_rechunked_${dataset}_${startyear}_${endyear}_noleap.nc4
echo 'rechunked' $i 'for faster access to full timeseries'

# OLD LOOP OVER ALL VARIABLES ROUTINE
# variables=( pr ps huss rhs rlds rsds tasmax tasmin tas wind )
# echo 'Rechunk the following variables'
# echo $variables
# for i in "${variables[@]}"; do
#     echo 'Rechunking' $i
#     nccopy -u -k 'nc4' -m 32G -c time/4018,lat/1,lon/720 ${datafolder}${i}_${dataset}_1901_2010_noleap.nc4 \
#         ${datafolder}${i}_rechunked_${dataset}_1901_2010_noleap.nc4
#     echo 'rechunked' $i 'for faster access to full timeseries'
# done
