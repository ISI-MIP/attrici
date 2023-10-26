#!/bin/bash

#SBATCH --qos=priority  
#SBATCH --partition=priority 
#SBATCH --job-name=create_ssa_smoothed_gmt
#SBATCH --account=dmcci
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

module purge
module load anaconda/2021.11

source ./variables_for_shellscripts.sh

$attrici_python_gmt calc_gmt_by_ssa.py



