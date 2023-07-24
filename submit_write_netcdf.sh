#!/bin/bash

# copy this to your run folder, and replace runid with the
# name of your run at all locations. Replace also --mail-user

#SBATCH --job-name=runid_merge
#SBATCH --account=isimip
#SBATCH --output=/p/tmp/sitreu/log/attrici/%x/%A_%a.log
#SBATCH --error=/p/tmp/sitreu/log/attrici/%x/%A_%a.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sitreu@pik-potsdam.de

# block one node to have enough memory
#SBATCH --partition=largemem
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

/home/sitreu/.conda/envs/attrici_2/bin/python -u write_netcdf.py
