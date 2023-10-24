#!/bin/bash

#SBATCH --qos=short 
#SBATCH --partition=standard
#SBATCH --job-name=attrici_rm_ts_traces
#SBATCH --account=dmcci
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sitreu@pik-potsdam.de
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00 


# find . -name "traces" -type d

tile=00010
source ./variables_for_shellscripts.sh

cd $project_basedir/output/${tile}/
for var in tas0 tas6 tas12 tas18 pr0 pr6 pr12 pr18 tas tasrange rsds
do
  rm -rf ./attrici_03_era5_t${tile}_${var}_rechunked/timeseries
done


echo "deleted successfully"




