#!/bin/bash

#SBATCH --qos=short
#SBATCH --partition=standard
#SBATCH --job-name=attrici_merge_parameters
#SBATCH --account=dmcci
#SBATCH --mail-type=END,FAIL
#SBATCH --cpus-per-task=6
#SBATCH --time=23:30:00

source ./variables_for_shellscripts.sh

$attrici_python -u ./merge_parameter_files.py
