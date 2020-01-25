#!/bin/bash

# copy this to your run folder, and replace runid with the
# name of your run at all locations. Replace also --mail-user

#SBATCH --qos=priority
#SBATCH --partition=priority
#SBATCH --job-name=runid_merge
#SBATCH --account=isipedia
#SBATCH --output=./log/%x.out
#SBATCH --error=./log/%x.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mengel@pik-potsdam.de

# block one node to have enough memory
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

/p/tmp/mengel/condaenvs/isi-cfact/bin/python -u write_netcdf.py
