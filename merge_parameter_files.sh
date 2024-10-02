#!/bin/bash

#SBATCH --qos=short
#SBATCH --partition=standard
#SBATCH --job-name=attrici_merge_parameters
#SBATCH --account=dmcci
#SBATCH --mail-type=END,FAIL
#SBATCH --cpus-per-task=6
#SBATCH --time=23:30:00

module purge
module load singularity

source ./variables_for_shellscripts.sh

tmpdir=$project_basedir/singularity_tmp/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p $tmpdir
export SINGULARITY_TMPDIR=$tmpdir

cleanup() {
  rm -rf ${tmpdir}
  exit
}

trap cleanup SIGTERM SIGINT

singularity exec -B /p:/p ${project_basedir}/ATTRICI.sif bash -c "cd $RUNDIR; python -u merge_parameter_files.py"
cleanup
echo "finished merging the parameter files"
