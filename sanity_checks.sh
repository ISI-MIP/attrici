#!/bin/bash
#### sanity slurm ####

#SBATCH --qos=short #priority
#SBATCH --partition=standard #priority
#SBATCH --account=dmcci
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

module purge
module load singularity

source ./variables_for_shellscripts.sh

# $attrici_python sanity_check.py # return general failure if any assertion in python script fails 
singularity exec -B /p:/p ${project_basedir}/ATTRICI.sif bash -c "cd $RUNDIR; python -u sanity_check.py || exit 1"

echo "Finished checks"
