#!/bin/bash

#SBATCH --qos=priority
#SBATCH --partition=priority
#SBATCH --job-name=complete_pre
#SBATCH --account=isipedia
#SBATCH --output=../output/%x.out
#SBATCH --error=../output/%x.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bschmidt@pik-potsdam.de
#SBATCH --time=00-23:50:00

# # block one node completely to get all its memory.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60000
#SBATCH --exclusive

module purge
module load anaconda/5.0.0_py3
module load intel/2019.4
module load netcdf-c/4.6.1/intel/parallel
module load nco/4.7.8

# Run complete preprocessing routine
echo "-----\n"
if [ "$USER" == "bschmidt" ]; then
    echo "\nUser is\n" $USER
    source activate mpi_py3
fi
echo "\nPython interpreter used is:"
which python
echo "\n-----\n"

srun yes | bash merge_data.sh
srun bash rechunk.sh
# srun /home/bschmidt/.conda/envs/mpi_py3/bin/python rechunk.py
# srun bash ncks_subset.sh
srun python subset.py
# python3 calc_gmt_by_ssa.py
