#!/bin/bash

#SBATCH --qos=priority
##SBATCH --partition=priority
#SBATCH --job-name=rechunk
#SBATCH --account=isipedia
#SBATCH --output=../output/%x.out
#SBATCH --error=../output/%x.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bschmidt@pik-potsdam.de

# # block one node completely to get all its memory.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive

# Run complete preprocessing routine
if [ "$USER" == "bschmidt" ]; then
    echo 'User is' $USER
    source /home/bschmidt/.programs/anaconda3/bin/activate detrending
fi

srun yes | bash merge_data.sh
# srun bash remove_leap_days.sh
srun bash rechunk.sh
srun python3 create_test_data.py
# python3 calc_gmt_by_ssa.py
