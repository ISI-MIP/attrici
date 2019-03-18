#!/bin/bash

#SBATCH --qos=priority
##SBATCH --partition=priority
#SBATCH --job-name=rechunk
#SBATCH --account=isipedia
#SBATCH --output=output/rechunk.out
#SBATCH --error=output/rechunk.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bschmidt@pik-potsdam.de

# # block one node completely to get all its memory.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#

nccopy -u -k 'nc4' -d0 -m 32M -c time/,lat/1,lon/ /home/bschmidt/temp/gswp3/pr_rm_gswp3_1901_2010.nc4 /home/bschmidt/temp/gswp3/pr_rm_rechunked_gswp3_1901_2010.nc4
nccopy -u -k 'nc4' -d0 -m 32M -c time/,lat/1,lon/ /home/bschmidt/temp/gswp3/ps_rm_gswp3_1901_2010.nc4 /home/bschmidt/temp/gswp3/ps_rm_rechunked_gswp3_1901_2010.nc4
nccopy -u -k 'nc4' -d0 -m 32M -c time/,lat/1,lon/ /home/bschmidt/temp/gswp3/huss_rm_gswp3_1901_2010.nc4 /home/bschmidt/temp/gswp3/huss_rm_rechunked_gswp3_1901_2010.nc4
nccopy -u -k 'nc4' -d0 -m 32M -c time/,lat/1,lon/ /home/bschmidt/temp/gswp3/rhs_rm_gswp3_1901_2010.nc4 /home/bschmidt/temp/gswp3/rhs_rm_rechunked_gswp3_1901_2010.nc4
nccopy -u -k 'nc4' -d0 -m 32M -c time/,lat/1,lon/ /home/bschmidt/temp/gswp3/rlds_rm_gswp3_1901_2010.nc4 /home/bschmidt/temp/gswp3/rlds_rm_rechunked_gswp3_1901_2010.nc4
nccopy -u -k 'nc4' -d0 -m 32M -c time/,lat/1,lon/ /home/bschmidt/temp/gswp3/rsds_rm_gswp3_1901_2010.nc4 /home/bschmidt/temp/gswp3/rsds_rm_rechunked_gswp3_1901_2010.nc4
nccopy -u -k 'nc4' -d0 -m 32M -c time/,lat/1,lon/ /home/bschmidt/temp/gswp3/tasmax_rm_gswp3_1901_2010.nc4 /home/bschmidt/temp/gswp3/tasmax_rm_rechunked_gswp3_1901_2010.nc4
nccopy -u -k 'nc4' -d0 -m 32M -c time/,lat/1,lon/ /home/bschmidt/temp/gswp3/tasmin_rm_gswp3_1901_2010.nc4 /home/bschmidt/temp/gswp3/tasmin_rm_rechunked_gswp3_1901_2010.nc4
nccopy -u -k 'nc4' -d0 -m 32M -c time/,lat/1,lon/ /home/bschmidt/temp/gswp3/tas_rm_gswp3_1901_2010.nc4 /home/bschmidt/temp/gswp3/tas_rm_rechunked_gswp3_1901_2010.nc4
