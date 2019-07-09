#!/bin/bash

#SBATCH --qos=short
#SBATCH --partition=ram_gpu
##SBATCH --constraint=broadwell
#SBATCH --job-name=ram_gpu
#SBATCH --account=isipedia
#SBATCH --output=../output/ram.out
#SBATCH --error=../output/ram.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bschmidt@pik-potsdam.de

# # block one node completely to get all its memory.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
##SBATCH --mem=70000
#SBATCH --exclusive

# Minimal script for extracting a regularly spaced subset from a larger dataset
module load nco/4.7.8
module load netcdf-c/4.6.1/intel/parallel
module load intel/2019.4

sub=5

in=/home/bschmidt/temp/gswp3/input/tasmin_gswp3.nc4
out=/home/bschmidt/temp/gswp3/input/tasmin_gswp3_2p5deg.nc4
# ncks -d lon,,,$sub -d lat,,,$sub $in $out
ncks --diskless_all -d lon,,,$sub -d lat,,,$sub $in $out
