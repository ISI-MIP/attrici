#!/bin/bash

#SBATCH --qos=priority
#SBATCH --partition=priority
#SBATCH --job-name=prsn_pp
#SBATCH --account=isipedia
#SBATCH --output=../output/%x.out
#SBATCH --error=../output/%x.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bschmidt@pik-potsdam.de
#SBATCH --time=00-23:59:59
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
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

pr=${datafolder}/input/pr_${dataset}.nc4
prsn=${datafolder}/input/prsn_${dataset}.nc4
prsnratio=${datafolder}/input/prsnratio_${dataset}.nc4

echo "Creating prsnratio"
echo 'Inputfiles:' ${pr} ${prsn} 
echo 'Outputfile:' ${prsnratio} 

cdo -O chname,prsn,prsnratio -div ${prsn} ${pr} temp.nc4
# if precipitation greater than threshold (1mm/day), then use prsnvalue from temp.nc4
# else go with pr-value (close to zero). This inhibits creation of NaN's by zero-division
# TODO: find a cleaner solution for this or set all small values to 0. But this will also be caught by the threshold in main algorithm
cdo -O ifthenelse -gtc,0.0000011574 ${pr} temp.nc4 ${pr} temp2.nc4 
rm temp.nc4

if [ $? == 0 ]; then
    echo "Finished without error"
else
    echo "Ups. Something went wrong"
fi
