#!/bin/bash

#SBATCH --qos=priority
#SBATCH --partition=priority
#SBATCH --job-name=complete_pre
#SBATCH --account=isipedia
#SBATCH --output=../output/%x.out
#SBATCH --error=../output/%x.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bschmidt@pik-potsdam.de
#SBATCH --time=00-23:59:59

# # block one node completely to get all its memory.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
##SBATCH --cpus-per-task=16
#SBATCH --mem=60000
#SBATCH --exclusive

module purge
# module load anaconda/5.0.0_py3
module load intel/2018.1
module load netcdf-c/4.6.1/intel/serial
module load nco/4.7.8

echo "Begin merging!"
yes | bash merge_data.sh
echo "Done merging!"
echo "Begin rechunking!"
bash rechunk.sh
echo "Done rechunking!"
echo "Begin subsetting!"
bash subset.sh
echo "Done subsetting!"
# python3 calc_gmt_by_ssa.py
