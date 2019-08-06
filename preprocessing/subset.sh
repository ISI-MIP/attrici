#!/bin/bash

#SBATCH --qos=priority
#SBATCH --partition=priority
#SBATCH --job-name=subset
#SBATCH --account=isipedia
#SBATCH --output=../output/%x.out
#SBATCH --error=../output/%x.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bschmidt@pik-potsdam.de
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=60000
#SBATCH --exclusive

# Minimal script for extracting a regularly spaced subset from a larger dataset
module purge
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
sub="$(grep 'lateral_sub =' ${settings_file} | cut -d' ' -f3 | sed "s/'//g" | sed 's/"//g')"

in=${datafolder}/input/${variable}_${dataset}.nc4
out=${datafolder}/input/${variable}_${dataset}_sub.nc4

echo $variable
echo 'Inputfile:' ${in}
echo 'Outputfile:' ${out}

cdo samplegrid,$sub ${in} ${out}
echo 'created subset for ' $variable ' with lateral step: ' $sub
