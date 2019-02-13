#!/bin/bash

#SBATCH --qos=priority
##SBATCH --partition=priority
#SBATCH --job-name=dtr_gmt
#SBATCH --account=isipedia
#SBATCH --output=output/dtr_gmt.out
#SBATCH --error=output/dtr_gmt.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bschmidt@pik-potsdam.de

# # block one node completely to get all its memory.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive

source /home/bschmidt/.programs/anaconda3/bin/activate detrending
srun python3 gmt.py
