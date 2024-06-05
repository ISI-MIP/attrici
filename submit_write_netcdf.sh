#!/bin/bash

# copy this to your run folder, and replace runid with the
# name of your run at all locations. Replace also --mail-user

#SBATCH --job-name=runid_merge
#SBATCH --account=dmcci
#SBATCH --mail-type=END,FAIL
##SBATCH --partition=largemem
#SBATCH --qos=priority
#SBATCH --partition=priority
#SBATCH --ntasks=1
##SBATCH --cpus-per-task=8
#SBATCH --mem=500M
#SBATCH --time=23:59:00

module purge
module load singularity

source ./variables_for_shellscripts.sh
export RUNDIR=$(pwd)
singularity exec -B /p:/p ${project_basedir}/ATTRICI.sif bash -c "cd $RUNDIR; python -u write_netcdf.py"
# $attrici_python -u write_netcdf.py
