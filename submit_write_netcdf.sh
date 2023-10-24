#!/bin/bash

# copy this to your run folder, and replace runid with the
# name of your run at all locations. Replace also --mail-user

#SBATCH --job-name=runid_merge
#SBATCH --account=dmcci
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=largemem
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=23:59:00

source ./variables_for_shellscripts.sh
attrici_python -u write_netcdf.py
