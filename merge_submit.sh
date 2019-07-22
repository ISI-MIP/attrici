#!/bin/bash

#SBATCH --qos=priority
#SBATCH --partition=priority
#SBATCH --job-name=merge
#SBATCH --account=isipedia
#SBATCH --output=test.out
#SBATCH --error=test.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bschmidt@pik-potsdam.de
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=1
##SBATCH --time=00-00:55:00

module purge
module load anaconda/5.0.0_py3
module load intel/2019.4
module load netcdf-c/4.6.2/intel/parallel
module load hdf5/1.10.2/intel/parallel

# export H5DIR=/p/system/packages/hdf5/1.8.18/impi
# export NCDIR=/p/system/packages/netcdf-c/4.4.1.1/impi

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/bschmidt/.conda/envs/mpi_py3/lib/libfabric/libfabric.so
export FI_PROVIDER_PATH=/home/bschmidt/.conda/envs/mpi_py3/lib/libfabric/prov
export I_MPI_FABRICS=shm:ofi # shm:dapl not applicable for libfabric
unset I_MPI_DAPL_UD
unset I_MPI_DAPL_UD_PROVIDER
export I_MPI_PMI_LIBRARY=/p/system/slurm/lib/libpmi.so
export SUBMITTED=1

srun -n $SLURM_NTASKS /home/bschmidt/.conda/envs/par_io/bin/python -u merge_parallel.py

echo "Finished run."
