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
if [ -e settings.py ]; then
    settings_file=settings.py 
else 
    settings_file=../settings.py
fi 

# Get information from settings file
# sed gets rid of string markers (',")
variable="$(grep 'variable =' ${settings_file} | cut -d' ' -f3 | sed "s/'//g" | sed 's/"//g')"
# datafolder selections relies on the folder being wrapped in double quotation marks
datafolder="$(grep 'data_dir =' ${settings_file} | grep $USER | cut -d'"' -f2 | sed "s/'//g" | sed 's/"//g')"

dataset="$(grep 'dataset =' ${settings_file} | cut -d' ' -f3 | sed "s/'//g" | sed 's/"//g')"
startyear="$(grep 'startyear =' ${settings_file} | cut -d' ' -f3 | sed "s/'//g" | sed 's/"//g')"
endyear="$(grep 'endyear =' ${settings_file} | cut -d' ' -f3 | sed "s/'//g" | sed 's/"//g')"

echo 'Rechunk the following variable'
echo $variable
echo 'Inputfile:' ${datafolder}${variable}_${dataset}_${startyear}_${endyear}_noleap.nc4
echo 'Outputfile:' ${datafolder}${variable}_rechunked_${dataset}_${startyear}_${endyear}_noleap.nc4
nccopy -u -k 'nc4' -m 32G -c time/4018,lat/1,lon/720 ${datafolder}${variable}_${dataset}_${startyear}_${endyear}_noleap.nc4 \
    ${datafolder}${variable}_rechunked_${dataset}_${startyear}_${endyear}_noleap.nc4
echo 'rechunked' $i 'for faster access to full timeseries'
