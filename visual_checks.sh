#!/bin/bash

#SBATCH --qos=short
#SBATCH --partition=standard
#SBATCH --job-name=attrici_visualcheck
#SBATCH --account=dmcci
#SBATCH --mail-type=END,FAIL
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00

source ./variables_for_shellscripts.sh

$attrici_python -u ./visual_check.py
