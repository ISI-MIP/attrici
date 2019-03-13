#!/bin/bash

#SBATCH --qos=priority
##SBATCH --partition=priority
#SBATCH --job-name=tas_regr
#SBATCH --account=isipedia
#SBATCH --output=output/tas_regr.out
#SBATCH --error=output/tas_regr.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bschmidt@pik-potsdam.de

# # block one node completely to get all its memory.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
echo 'Available memory of node is:'
cat /proc/meminfo | grep MemFree | awk '{ print $2 }'
source /home/bschmidt/.programs/anaconda3/bin/activate detrending
srun python3 iris_regr.py
# run next line for profiling memory
# srun mprof run --include-children --multiprocess iris_regr.py
