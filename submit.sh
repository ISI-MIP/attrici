#!/bin/bash
#
#SBATCH --qos=priority
#SBATCH --partition=priority
##SBATCH --constraint=broadwell
#SBATCH --job-name=vergessen
#SBATCH --account=isipedia
#SBATCH --output=output/%x.out
#SBATCH --error=output/%x.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bschmidt@pik-potsdam.de

# # block one node completely to get all its memory.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
##SBATCH --cpus-per-task=16
##SBATCH --exclusive
# echo 'Available memory of node is:'
# cat /proc/meminfo | grep MemFree | awk '{ print $2 }'
# source /home/bschmidt/.programs/anaconda3/bin/activate detrending_idp

modul purge
module load anaconda/5.0.0_py3
source activate mpi_py3
which python3

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/bschmidt/.conda/envs/mpi_py3/lib/libfabric/libfabric.so
export FI_PROVIDER_PATH=/home/bschmidt/.conda/envs/mpi_py3/lib/libfabric/prov
export I_MPI_FABRICS=shm:ofi # shm:dapl not applicable for libfabric
unset I_MPI_DAPL_UD
unset I_MPI_DAPL_UD_PROVIDER
export I_MPI_PMI_LIBRARY=/p/system/slurm/lib/libpmi.so
# export I_MPI_DEBUG=5

# srun bash preprocessing/merge_data.sh
# srun bash preprocessing/remove_leap_days.sh
# srun python3 preprocessing/create_test_data.py
# srun python3 run_regression.py
# srun python3 run_detrending.py
# run next line for profiling memory
# srun mprof run --include-children --multiprocess run_regression.py

echo "Number of processes started:"
echo $SLURM_NPROCS
# srun -n $SLURM_NTASKS /home/bschmidt/.conda/envs/mpi_py3/bin/python3 -m mpi4py.futures run_bayes_reg.py
srun -n $SLURM_NTASKS /home/bschmidt/.conda/envs/mpi_py3/bin/python3 -m mpi4py.futures run_bayes_reg.py

echo "executed mpi"
