#!/bin/bash

#SBATCH --qos=priority
#SBATCH --partition=priority
#SBATCH --job-name=merge_cfact
#SBATCH --account=isipedia
#SBATCH --output=../output/%x.out
#SBATCH --error=../output/%x.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bschmidt@pik-potsdam.de
#SBATCH --ntasks=1
#SBATCH --exclusive
##SBATCH --cpus-per-task=1

module purge
module load anaconda/5.0.0_py3

srun -n $SLURM_NTASKS /home/bschmidt/.conda/envs/mpi_py3/bin/python -u merge_cfact.py

echo "Finished run."
