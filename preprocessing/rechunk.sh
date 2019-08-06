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
module load nco/4.7.8
preprocessing=$1
# module load cdo/1.9.6/gnu-threadsafe

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

if [[ $preprocessing = 1 ]]; then
    inputfile=${datafolder}/input/${variable}_${dataset}.nc4
    outputfile=${datafolder}/input/${variable}_${dataset}_re.nc4
    echo 'Rechunk the following variable to be optimised for timeseries access'
    echo $variable
    echo 'Inputfile:' ${inputfile}
    echo 'Outputfile:' ${outputfile}
    ncks -4 -O -L 0 --cnk_csh=45000000000 --cnk_plc=g3d --cnk_dmn=time,42369 --cnk_dmn=lat,1 --cnk_dmn=lon,1 ${inputfile} ${outputfile}
else
    inputfile=${datafolder}/output/tas/cfact/${variable}_${dataset}_cfactual.nc4
    outputfile=${datafolder}/output/tas/cfact/${variable}_${dataset}_re.nc4
    echo 'Rechunk the following variable to be optimised for access of spatial fields and one time slice.'
    echo $variable
    echo 'Inputfile:' ${inputfile}
    echo 'Outputfile:' ${outputfile}
    ncks -4 -O -L 0 --cnk_csh=45000000000 --cnk_plc=g3d --cnk_dmn=time,1 --cnk_dmn=lat,360 --cnk_dmn=lon,720 ${inputfile} ${outputfile}
fi

echo 'rechunked' $variable 'for faster access to full timeseries'
