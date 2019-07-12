#!/bin/bash

#SBATCH --qos=priority
#SBATCH --partition=ram_gpu
#SBATCH --job-name=rechunk
#SBATCH --account=isipedia
#SBATCH --output=../output/%x.out
#SBATCH --error=../output/%x.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bschmidt@pik-potsdam.de
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
##SBATCH --mem=60000
#SBATCH --exclusive

module load intel/2019.4
module load netcdf-c/4.6.1/intel/parallel
module load nco/4.7.8

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
# startyear="$(grep 'startyear =' ${settings_file} | cut -d' ' -f3 | sed "s/'//g" | sed 's/"//g')"
# endyear="$(grep 'endyear =' ${settings_file} | cut -d' ' -f3 | sed "s/'//g" | sed 's/"//g')"

inputfile=${datafolder}/input/${variable}_${dataset}.nc4
outputfile=${datafolder}/input/${variable}_${dataset}_re.nc4
# echo 'Rechunk the following variable'
# echo $variable
echo 'Inputfile:' ${inputfile}
echo 'Outputfile:' ${outputfile}
# nccopy -w -k 'nc4' -c time/4018,lat/1,lon/720 ${inputfile} temp_${variable}.nc4
# nccopy -w -k 'nc4' -c time/40177,lat/1,lon/1 temp_${variable}.nc4 ${outputfile}
nccopy -w -k 'nc4' -c time/40177,lat/1,lon/1 ${inputfile} ${outputfile}
# rm temp_${variable}.nc4
echo 'rechunked' $variable 'for faster access to full timeseries'
