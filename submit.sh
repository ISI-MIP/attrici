#!/bin/bash

#SBATCH --qos=priority
#SBATCH --partition=priority
##SBATCH --constraint=broadwell
#SBATCH --job-name=test
#SBATCH --account=isipedia
#SBATCH --output=output/test.out
#SBATCH --error=output/test.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bschmidt@pik-potsdam.de

# # block one node completely to get all its memory.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
echo 'Available memory of node is:'
cat /proc/meminfo | grep MemFree | awk '{ print $2 }'
source /home/bschmidt/.programs/anaconda3/bin/activate detrending
srun python3 run_regression_classic.py
# run next line for profiling memory
# srun mprof run --include-children --multiprocess run_regression_classic.py
