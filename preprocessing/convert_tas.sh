#!/bin/bash

#SBATCH --qos=priority
#SBATCH --partition=priority
#SBATCH --job-name=rechunk
#SBATCH --account=isipedia
#SBATCH --output=../output/%x.out
#SBATCH --error=../output/%x.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bschmidt@pik-potsdam.de
#SBATCH --time=00-23:59:59
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
##SBATCH --cpus-per-task=16
#SBATCH --mem=60000
#SBATCH --exclusive

module purge
module load intel/2018.1
module load netcdf-c/4.6.1/intel/serial
module load cdo/1.9.6/gnu-threadsafe


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

tas=${datafolder}/input/tas_${dataset}_sub.nc4
tasmax=${datafolder}/input/tasmax_${dataset}_sub.nc4
tasmin=${datafolder}/input/tasmin_${dataset}_sub.nc4
tasrange=${datafolder}/input/tasrange_${dataset}_sub.nc4
tasskew=${datafolder}/input/tasskew_${dataset}_sub.nc4
tasskewtemp=${datafolder}/input/tasskew_${dataset}_sub.nc4temp

echo "Creating tasrange and tasskew"
echo 'Inputfiles:' ${tas} ${tasmax} ${tasmin}
echo 'Outputfile:' ${tasrange} ${tasskew}

cdo -O chname,tasmax,tasrange -sub $tasmax $tasmin $tasrange
cdo -O sub $tas $tasmin $tasskewtemp
cdo -O chname,tas,tasskew -div $tasskewtemp $tasrange $tasskew 
rm $tasskewtemp

if [ $? == 0 ]; then
    echo "Finished without error"
else
    echo "Ups. Something went wrong"
fi
