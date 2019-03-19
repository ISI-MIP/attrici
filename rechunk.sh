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

cdo delete,month=2,day=29 /home/bschmidt/temp/gswp3/pr_rm_gswp3_1901_2010.nc4 temp.nc
mv temp.nc /home/bschmidt/temp/gswp3/pr_rm_gswp3_1901_2010.nc4
echo 'Removed leap day from pr'
cdo delete,month=2,day=29 /home/bschmidt/temp/gswp3/ps_rm_gswp3_1901_2010.nc4 temp.nc
mv temp.nc /home/bschmidt/temp/gswp3/ps_rm_gswp3_1901_2010.nc4
echo 'Removed leap day from ps'
cdo delete,month=2,day=29 /home/bschmidt/temp/gswp3/huss_rm_gswp3_1901_2010.nc4 temp.nc
mv temp.nc /home/bschmidt/temp/gswp3/huss_rm_gswp3_1901_2010.nc4
echo 'Removed leap day from huss'
cdo delete,month=2,day=29 /home/bschmidt/temp/gswp3/rhs_rm_gswp3_1901_2010.nc4 temp.nc
mv temp.nc /home/bschmidt/temp/gswp3/rhs_rm_gswp3_1901_2010.nc4
echo 'Removed leap day from rhs'
cdo delete,month=2,day=29 /home/bschmidt/temp/gswp3/rlds_rm_gswp3_1901_2010.nc4 temp.nc
mv temp.nc /home/bschmidt/temp/gswp3/rlds_rm_gswp3_1901_2010.nc4
echo 'Removed leap day from rlds'
cdo delete,month=2,day=29 /home/bschmidt/temp/gswp3/rsds_rm_gswp3_1901_2010.nc4 temp.nc
mv temp.nc /home/bschmidt/temp/gswp3/rsds_rm_gswp3_1901_2010.nc4
echo 'Removed leap day from rsds'
cdo delete,month=2,day=29 /home/bschmidt/temp/gswp3/tasmax_rm_gswp3_1901_2010.nc4 temp.nc
mv temp.nc /home/bschmidt/temp/gswp3/tasmax_rm_gswp3_1901_2010.nc4
echo 'Removed leap day from tasmax'
cdo delete,month=2,day=29 /home/bschmidt/temp/gswp3/tasmin_rm_gswp3_1901_2010.nc4 temp.nc
mv temp.nc /home/bschmidt/temp/gswp3/tasmin_rm_gswp3_1901_2010.nc4
echo 'Removed leap day from tasmin'
cdo delete,month=2,day=29 /home/bschmidt/temp/gswp3/tas_rm_gswp3_1901_2010.nc4 temp.nc
mv temp.nc /home/bschmidt/temp/gswp3/tas_rm_gswp3_1901_2010.nc4
echo 'Removed leap day from tas'

nccopy -u -k 'nc4' -m 32G -c time/4018,lat/1,lon/720 /home/bschmidt/temp/gswp3/pr_rm_gswp3_1901_2010.nc4 /home/bschmidt/temp/gswp3/pr_rm_rechunked_gswp3_1901_2010.nc4
echo 'rechunked pr'
nccopy -u -k 'nc4' -m 32G -c time/4018,lat/1,lon/720 /home/bschmidt/temp/gswp3/ps_rm_gswp3_1901_2010.nc4 /home/bschmidt/temp/gswp3/ps_rm_rechunked_gswp3_1901_2010.nc4
echo 'rechunked ps'
nccopy -u -k 'nc4' -m 32G -c time/4018,lat/1,lon/720 /home/bschmidt/temp/gswp3/huss_rm_gswp3_1901_2010.nc4 /home/bschmidt/temp/gswp3/huss_rm_rechunked_gswp3_1901_2010.nc4
echo 'rechunked huss'
nccopy -u -k 'nc4' -m 32G -c time/4018,lat/1,lon/720 /home/bschmidt/temp/gswp3/rhs_rm_gswp3_1901_2010.nc4 /home/bschmidt/temp/gswp3/rhs_rm_rechunked_gswp3_1901_2010.nc4
echo 'rechunked rhs'
nccopy -u -k 'nc4' -m 32G -c time/4018,lat/1,lon/720 /home/bschmidt/temp/gswp3/rlds_rm_gswp3_1901_2010.nc4 /home/bschmidt/temp/gswp3/rlds_rm_rechunked_gswp3_1901_2010.nc4
echo 'rechunked rlds'
nccopy -u -k 'nc4' -m 32G -c time/4018,lat/1,lon/720 /home/bschmidt/temp/gswp3/rsds_rm_gswp3_1901_2010.nc4 /home/bschmidt/temp/gswp3/rsds_rm_rechunked_gswp3_1901_2010.nc4
echo 'rechunked rsds'
nccopy -u -k 'nc4' -m 32G -c time/4018,lat/1,lon/720 /home/bschmidt/temp/gswp3/tasmax_rm_gswp3_1901_2010.nc4 /home/bschmidt/temp/gswp3/tasmax_rm_rechunked_gswp3_1901_2010.nc4
echo 'rechunked tasmax'
nccopy -u -k 'nc4' -m 32G -c time/4018,lat/1,lon/720 /home/bschmidt/temp/gswp3/tasmin_rm_gswp3_1901_2010.nc4 /home/bschmidt/temp/gswp3/tasmin_rm_rechunked_gswp3_1901_2010.nc4
echo 'rechunked tasmin'
nccopy -u -k 'nc4' -m 32G -c time/4018,lat/1,lon/720 /home/bschmidt/temp/gswp3/tas_rm_gswp3_1901_2010.nc4 /home/bschmidt/temp/gswp3/tas_rm_rechunked_gswp3_1901_2010.nc4
echo 'rechunked tas'
