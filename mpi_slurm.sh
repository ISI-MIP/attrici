#!/bin/bash

#SBATCH --qos=short #standby   #short
#SBATCH --partition=standard #priority
#SBATCH --job-name=attrici_mpi_run_estimation
#SBATCH --account=dmcci 
#SBATCH --nodes=1   # 4 with 1 task-per-node= one var processed by 4 nodes at same time
#SBATCH --ntasks-per-node=2  # 1 = only one var is process - probably as threating process
#SBATCH --array=0-100
#SBATCH --cpus-per-task=8
#SBATCH --output=/p/tmp/annabu/projects/attrici/log/mpi/run_estimation_%A_%a_mpi-%j.out
#SBATCH --output=/p/tmp/annabu/projects/attrici/log/mpi/run_estimation_%A_%a_mpi-%j.out
#SBATCH --time=00-06:00:00

export CXX=g++
tmpdir=/p/tmp/annabu/projects/attrici/tmp/theano_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${SLURM_NTASKS}.tmp
mkdir -p $tmpdir
export TMPDIR=$tmpdir

## test if uncomment or modification needed
## if you're using OpenMP for threading
# unset I_MPI_DAPL_UD
# unset I_MPI_DAPL_UD_PROVIDER

## srun version
export I_MPI_PMI_LIBRARY=/p/system/slurm/lib/libpmi.so

## test if uncomment or modification needed
# if you're using OpenMP for threading
export OMP_PROC_BIND=true # make sure our threads stick to cores
export OMP_NUM_THREADS=2  # matches how many cpus-per-task we asked for  # todo test change value or uncomment
export SUBMITTED=1

compiledir=/p/tmp/annabu/projects/attrici/.pytensor/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p $compiledir
export PYTENSOR_FLAGS=base_compiledir=$compiledir


echo "------------------------------------------------------------"
echo "SLURM JOB ID: $SLURM_JOBID"
echo "$SLURM_NTASKS tasks"
echo "------------------------------------------------------------"
# we can use "-n 64" (4*16) in the run commands below or the 
# shortcut $SLURM_NTASKS
# See http://slurm.schedmd.com/sbatch.html for a full list of 
# input and output variables


cleanup() {
  rm -r ${compiledir}
  rm -r ${tmpdir}
  exit
}
cp .pytensorrc /home/annabu/.pytensorrc
trap cleanup SIGTERM SIGINT

for var_folder in attrici_03_era5_t00002_sfcWind_rechunked attrici_03_era5_t00002_tas18_rechunked attrici_03_era5_t00002_rsds_rechunked # ; do
do
    echo ${var_folder}
    srun -n $SLURM_NTASKS /home/annabu/.conda/envs/attrici_pymc5_2/bin/python -u /p/tmp/annabu/projects/attrici/runscripts/attrici_automated_processing/00002/${var_folder}/run_estimation.py  #todo test with & 
done
wait


cleanup

## EXAMPLES 
#srun -n $SLURM_NTASKS ./sumprimes 0 1000000
# same py script for  multiple infiles
#for i in {1..100}  
#do
#    srun -n 1 python3 xxx.py -i /path/input$i &
#done
#wait


# alternatively, use mpirun directly. We have to have the modules
# loaded by this point.
# e.g. "module load intel/2017.1"
#mpirun -bootstrap slurm -n $SLURM_NTASKS \ 
#  $HOME/cluster-examples/sumprimes/mpi/sumprimes 0 10000000