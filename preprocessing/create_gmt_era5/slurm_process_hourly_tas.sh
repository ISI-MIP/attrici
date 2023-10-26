#!/bin/bash

#SBATCH --partition=priority
#SBATCH --qos=priority
#SBATCH --job-name=mean_computation
#SBATCH --output=/p/tmp/sitreu/log/attrici/create_era5_gmt/log_%A_%a.log
#SBATCH --error=/p/tmp/sitreu/log/attrici/create_era5_gmt/log_%A_%a.log
#SBATCH --account=dmcci
#SBATCH --array=1950-2020
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2


YEAR=${SLURM_ARRAY_TASK_ID}
sh process_hourly_tas_files.sh $YEAR

