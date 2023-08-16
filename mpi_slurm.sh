#!/bin/bash

#SBATCH --qos=short #priority standby 
#SBATCH --partition=standard ## standard #priority
#SBATCH --job-name=attrici_mpi_t2_run_estimation
#SBATCH --account=dmcci 
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=annabu@pik-potsdam.de
#SBATCH --nodes=1    
#SBATCH --ntasks-per-node=3
#SBATCH --array=0-100
#SBATCH --cpus-per-task=10  ## same as OMP_NUM_THEADS
#SBATCH --output=/p/tmp/annabu/projects/attrici/log/00002/run_estimation_%A_%a_mpi-%j.out
#SBATCH --error=/p/tmp/annabu/projects/attrici/log/00002/run_estimation_%A_%a_mpi-%j.out
#SBATCH --time=07:30:00  

## SLURM_NTASKS = nodes * ntaskes-per-node

export CXX=g++
tmpdir=/p/tmp/annabu/projects/attrici/tmp/theano_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${SLURM_NTASKS}.tmp
mkdir -p $tmpdir
export TMPDIR=$tmpdir



## test if uncomment or modification needed
## if you're using OpenMP for threading
#unset I_MPI_DAPL_UD   # TEST: as uncommented
#unset I_MPI_DAPL_UD_PROVIDER

## get mpi
export I_MPI_PMI_LIBRARY=/p/system/slurm/lib/libpmi.so

# if you're using OpenMP for threading
export OMP_PROC_BIND=true # make sure our threads stick to cores
export OMP_NUM_THREADS=10   # matches how many cpus-per-task we asked for  # todo test change value or uncomment
# OMP_NUM_THREADS: so that OpenMP knows how many threads to use per task.
export SUBMITTED=1
compiledir=/p/tmp/annabu/projects/attrici/.pytensor/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p $compiledir
export PYTENSOR_FLAGS=base_compiledir=$compiledir


echo "------------------------------------------------------------"
echo "SLURM JOB ID: $SLURM_JOBID"
echo "$SLURM_NTASKS tasks"
echo "------------------------------------------------------------"
# See http://slurm.schedmd.com/sbatch.html for a full list of 
# input and output variables


cleanup() {
  rm -r ${compiledir}
  rm -r ${tmpdir}
  exit
}

cp .pytensorrc /home/annabu/.pytensorrc
trap cleanup SIGTERM SIGINT

#for var_folder in attrici_03_era5_t00002_sfcWind_rechunked attrici_03_era5_t00002_tas18_rechunked  # works fine
for var_folder in attrici_03_era5_t00002_pr0_rechunked attrici_03_era5_t00002_tasskew_rechunked attrici_03_era5_t00002_hurs_rechunked;
do
    echo ${var_folder}
    srun --exclusive --ntasks 1  /home/annabu/.conda/envs/attrici_pymc5_2/bin/python -u /p/tmp/annabu/projects/attrici/runscripts/attrici_automated_processing/00002/${var_folder}/run_estimation.py  & # --exclusive --ntasks 1 : without srun would use full allocation for each variable
    
    #    srun -n $SLURM_NTASKS  /home/annabu/.conda/envs/attrici_pymc5_2/bin/python -u /p/tmp/annabu/projects/attrici/runscripts/attrici_automated_processing/00002/${var_folder}/run_estimation.py  #todo test with &   # SLURM_NTASKS=ntask-per-node
done
wait

cleanup
