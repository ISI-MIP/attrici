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

# Execute the sanity check inside the singularity container
singularity exec -B /p:/p ${project_basedir}/ATTRICI.sif bash -c "cd $RUNDIR; python -u sanity_check.py"
if [ $? -ne 0 ]; then
  echo "sanity_check.py failed"
  exit 1
fi

echo "Finished checks"
