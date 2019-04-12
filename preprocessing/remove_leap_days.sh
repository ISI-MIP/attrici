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
datafolder=/home/bschmidt/temp/gswp3/
dataset=gswp3
variables=( pr ps huss rhs rlds rsds tasmax tasmin tas wind )
echo 'Remove leap days for the following variables'
echo $variables
for i in "${variables[@]}"; do
    echo "Starting deletion of leap days (29th of feb) for variable" $i
    cdo delete,month=2,day=29 ${datafolder}${i}_rm_gswp3_1901_2010.nc4 temp.nc
    mv temp.nc ${datafolder}${i}_gswp3_1901_2010.nc4
    echo 'Removed leap day from' $i
done
