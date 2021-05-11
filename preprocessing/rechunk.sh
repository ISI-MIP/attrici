#!/bin/bash

# copy this to your run folder, and replace runid with the
# name of your run at all locations. Replace also --mail-user

#SBATCH --qos=priority
##SBATCH --qos=short
#SBATCH --partition=priority
##SBATCH --partition=standard
#SBATCH --job-name=rechunking_gswp3
#SBATCH --account=isimip
#SBATCH --output=./log/%x.log
#SBATCH --error=./log/%x.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sitreu@pik-potsdam.de

# block one node to have enough memory
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16


var=$1
ifile=/p/tmp/sitreu/isimip/isi-cfact/input/GSWP3-map_chunks/${var}_gswp3_sub1.nc4
ofile=../$(basename ${ifile})
n_times=$(ncks --trd -m -M $ifile | grep -E -i ": time, size =" | cut -f 7 -d ' ' | uniq)
ncks -O --cnk_csh=15000000000 --cnk_plc=g3d --cnk_dmn=time,$n_times --cnk_dmn=lat,10 --cnk_dmn=lon,10 $ifile $ofile
