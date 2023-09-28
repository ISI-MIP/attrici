#!/bin/bash

#SBATCH --qos=short
#SBATCH --partition=standard
#SBATCH --job-name=attrici_visualcheck
#SBATCH --account=dmcci
#SBATCH --output=./log/visual_check/%A.log
#SBATCH --error=./log/visual_check/%A.log
#SBATCH --mail-user=annabu@pik-potsdam.de
#SBATCH --mail-type=END,FAIL
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00


tile=$1
var=$2
#outfile_visual_check=/p/tmp/annabu/projects/attrici/log/attrici_03_era5_t${tile}_${var}_rechunked/visual_check_final_cfact.log

echo $tile $var
/home/annabu/.conda/envs/attrici_pymc5_2/bin/python -u ./sanity_check/visual_check.py ${tile} ${var}
