#!/bin/bash

#SBATCH --qos=standby
#SBATCH --partition=priority
#SBATCH --job-name=attrici_run_estimation
#SBATCH --account=isimip
#SBATCH --output=/p/tmp/sitreu/log/attrici/%x/run_estimation_%A_%a.log
#SBATCH --error=/p/tmp/sitreu/log/attrici/%x/run_estimation_%A_%a.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sitreu@pik-potsdam.de
#SBATCH --ntasks=1
#SBATCH --array=0-1358
##SBATCH --array=0-20
#SBATCH --cpus-per-task=2
#SBATCH --time=00-12:00:00

# module purge
# module load compiler/gnu/7.3.0
# module load anaconda/2021.11
# module load git
# module load anaconda/5.0.0_py3

export CXX=g++
tmpdir=/p/tmp/sitreu/data/attrici/tmp/theano_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.tmp
mkdir -p $tmpdir
export TMPDIR=$tmpdir

## for 1arr+1CPU -> comment
## if you're using OpenMP for threading
# unset I_MPI_DAPL_UD
# unset I_MPI_DAPL_UD_PROVIDER
# export I_MPI_PMI_LIBRARY=/p/system/slurm/lib/libpmi.so
# if you're using OpenMP for threading
export OMP_PROC_BIND=true # make sure our threads stick to cores
export OMP_NUM_THREADS=2  # matches how many cpus-per-task we asked for
export SUBMITTED=1
compiledir=/p/tmp/sitreu/data/attrici/.pytensor/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p $compiledir
export PYTENSOR_FLAGS=base_compiledir=$compiledir

cleanup() {
  rm -r $compiledir
  rm -r $tmpdir
  exit
}
cp .pytensorrc /home/sitreu/.pytensorrc
trap cleanup SIGTERM SIGINT
/home/sitreu/.conda/envs/attrici_2/bin/python -u run_estimation.py
cleanup

echo "Finished run."
