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
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=5
##SBATCH --cpus-per-task=16
##SBATCH --exclusive
echo 'Available memory of node is:'
cat /proc/meminfo | grep MemFree | awk '{ print $2 }'
source /home/bschmidt/.programs/anaconda3/bin/activate detrending_idp
module load compiler/intel/17.0.0
export I_MPI_PMI_LIBRARY=/p/system/packages/intel/parallel_studio_xe_2018_update1/compilers_and_libraries_2018.1.163/linux/mpi/intel64/lib/libmpi.so
FI_PROVIDER=psm2
module load compiler/llvm/6.0.0
module load compiler/gnu/7.3.0
# srun bash preprocessing/merge_data.sh
# srun bash preprocessing/remove_leap_days.sh
# srun python3 preprocessing/create_test_data.py
# srun python3 run_regression.py
# srun python3 run_detrending.py
# run next line for profiling memory
# srun mprof run --include-children --multiprocess run_regression.py

echo "Number of processes started:"
echo $SLURM_NPROCS
# srun -n $SLURM_NTASKS python3 run_bayes_reg.py
mpiexec.hydra -bootstrap slurm -n $SLURM_NTASKS python3 run_bayes_reg.py
echo "executed mpi"
