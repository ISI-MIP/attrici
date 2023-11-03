#!/bin/bash
#### sanity slurm ####

#SBATCH --qos=short #priority
#SBATCH --partition=standard #priority
#SBATCH --account=dmcci
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

source ./variables_for_shellscripts.sh

$attrici_python -u sanity_check.py || exit 1 # return general failure if any assertion in python script fails 

echo "Finished checks"
