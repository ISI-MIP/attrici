#!/bin/bash

#SBATCH --qos=short # for smaller tiles
#SBATCH --partition=standard   # for smaller tiles
#SBATCH --account=dmcci
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=07:00:00


tile=$1
var=$2

source ./variables_for_shellscripts.sh


# merge trace and ts files
attrici_python -u ./sanity_check/merge_files.py ${tile} ${var} || exit 1

echo "Finished, created backup files for " ${tile} ${var} 

