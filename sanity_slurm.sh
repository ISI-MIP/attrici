#!/bin/bash
#### sanity slurm  ####

tile=$1
variable_hour=$2

#SBATCH --qos=priority
#SBATCH --partition=priority
#SBATCH --job-name=attrici_sanity_t00001_tas0
#SBATCH --account=dmcci
#SBATCH --output=/p/tmp/annabu/projects/attrici/log/sanity_checks/run_estimation_t00001_tas0_%A.log
#SBATCH --error=/p/tmp/annabu/projects/attrici/log/sanity_checks/sanity_check_t00001_tas0_%A.log
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=annabu@pik-potsdam.de
#SBATCH --ntasks=1
#SBATCH --array=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00-00:30:00

# module purge
# module load compiler/gnu/7.3.0
# module load anaconda/2021.11
# module load git
# module load anaconda/5.0.0_py3

export CXX=g++
tmpdir=/p/tmp/annabu/projects/attrici/tmp/theano_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.tmp
mkdir -p $tmpdir
export TMPDIR=$tmpdir

## for 1arr+1CPU -> comment
export OMP_PROC_BIND=true # make sure our threads stick to cores
export OMP_NUM_THREADS=2  # matches how many cpus-per-task we asked for
export SUBMITTED=1
compiledir=/p/tmp/annabu/projects/attrici/.pytensor/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p $compiledir
export PYTENSOR_FLAGS=base_compiledir=$compiledir

cleanup() {
  rm -r $compiledir
  rm -r $tmpdir
  exit
}
#cp .pytensorrc /home/annabu/.pytensorrc
trap cleanup SIGTERM SIGINT
/home/annabu/.conda/envs/attrici_pymc5_2/bin/python -u sanity_check/sanity_check.py ${tile} ${variable_hour}
cleanup

echo "Finished run."
