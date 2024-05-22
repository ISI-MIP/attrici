#!/bin/bash

#SBATCH --qos=standby  
#SBATCH --partition=priority 
#SBATCH --job-name=attrici_run_estimation
#SBATCH --account=dmcci
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1
#SBATCH --array=0-300%100
#SBATCH --cpus-per-task=4
#SBATCH --time=23:00:00
#SBATCH --output=/p/tmp/sitreu/projects/attrici/log/attrici_pymc3_gswp3-w5e5_pr/run_estimation/%x_%A_%a
#SBATCH --error=/p/tmp/sitreu/projects/attrici/log/attrici_pymc3_gswp3-w5e5_pr/run_estimation/%x_%A_%a


module purge
module load singularity

# module load anaconda/5.0.0_py3

source ./variables_for_shellscripts.sh

tmpdir=$project_basedir/tmp/theano_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.tmp
mkdir -p $tmpdir
export TMPDIR=$tmpdir
if [ -n "$tmpdir" ] && [ -d "$tmpdir" ]; then
    rm -rf "$tmpdir"/*
else
    echo "Error: \$tmpdir is either empty or not a directory. Aborting deletion."
fi


## for 1arr+1CPU -> comment
## if you're using OpenMP for threading
# unset I_MPI_DAPL_UD
# unset I_MPI_DAPL_UD_PROVIDER
# export I_MPI_PMI_LIBRARY=/p/system/slurm/lib/libpmi.so
# if you're using OpenMP for threading
export OMP_PROC_BIND=true # make sure our threads stick to cores
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}    # matches how many cpus-per-task we asked for
export SUBMITTED=1
compiledir=$project_basedir/.theano/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p $compiledir
export THEANO_FLAGS=base_compiledir=$compiledir
if [ -n "$compiledir" ] && [ -d "$compiledir" ]; then
    rm -rf "$compiledir"/*
else
    echo "Error: \$compiledir is either empty or not a directory. Aborting deletion."
fi


cleanup() {
  rm -rf ${compiledir}
  rm -rf ${tmpdir}
  exit
}
cp config/theanorc $HOME/.tensorrc
trap cleanup SIGTERM SIGINT

export RUNDIR=$(pwd)

singularity exec -B /p:/p ${project_basedir}/ATTRICI.sif bash -c "cd $RUNDIR; python -u run_estimation.py"

# $attrici_python -u run_estimation.py 
cleanup


echo "Finished run."
