#!/bin/bash

#SBATCH --qos=priority
#SBATCH --partition=priority
#SBATCH --job-name=merge
#SBATCH --account=isipedia
#SBATCH --output=./output/%x.out
#SBATCH --error=./output/%x.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bschmidt@pik-potsdam.de
#SBATCH --ntasks=64
#SBATCH --cpus-per-task=1
#SBATCH --time=00-23:50:00

module purge
module load brotli/1.0.2
module load anaconda/5.0.0_py3
module load compiler/gnu/7.3.0
module load intel/2019.4

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/bschmidt/.conda/envs/mpi_py3/lib/libfabric/libfabric.so
export FI_PROVIDER_PATH=/home/bschmidt/.conda/envs/mpi_py3/lib/libfabric/prov
export I_MPI_FABRICS=shm:ofi # shm:dapl not applicable for libfabric
unset I_MPI_DAPL_UD
unset I_MPI_DAPL_UD_PROVIDER
export I_MPI_PMI_LIBRARY=/p/system/slurm/lib/libpmi.so
export SUBMITTED=1

error_note() {
  echo "Ups. Something went wrong."
  exit
}

trap error_note SIGTERM
srun -n $SLURM_NTASKS /home/bschmidt/.conda/envs/isi-cfact/bin/python -u merge_parallel.py
