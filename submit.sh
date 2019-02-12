#!/bin/bash

#SBATCH --qos=short
#SBATCH --job-name=dtr_wind_smooth
#SBATCH --account=
#SBATCH --output=output/dtr_wind_smooth.out
#SBATCH --error=output/dtr_wind_smooth.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user={{user}}@pik-potsdam.de

# # block one node completely to get all its memory.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive

source /home/bschmidt/.programs/anaconda3/bin/activate py2a
srun python3 wind_smooth.py
