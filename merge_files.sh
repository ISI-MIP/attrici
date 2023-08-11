#!/bin/bash
#### merge files after passed sanity checks  ####

tile=$1
trace_or_ts=$2

##SBATCH --qos=standby # for smaller tiles
##SBATCH --partition=priority  # for smaller tiles
#SBATCH --partition=largemem  # for large tiles
#SBATCH --job-name=attrici_merge_files
#SBATCH --account=dmcci
#SBATCH --output=/p/tmp/annabu/projects/attrici/log/create_backups/merge_files_%A.log
#SBATCH --error=/p/tmp/annabu/projects/attrici/log/create_backups/merge_files_%A.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=annabu@pik-potsdam.de
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
##SBATCH --time=00-06:00:00



# merge trace and ts files for each variable after they passed sanity checks
#for var in pr0; do  
#for var in tas0 tas6 tas12 tasrange pr0 pr6 pr12 pr18 sfcWind rsds hurs; do  
for var in tas0 tas6 tas12 tasrange tasskew pr0 pr6 pr12 pr18 sfcWind rsds hurs; do  
  echo "Merging: " ${var}
  /home/annabu/.conda/envs/attrici_pymc5_2/bin/python -u sanity_check/merge_files.py ${tile} ${trace_or_ts} ${var}
done

echo "Finished, merged all" ${trace_or_ts}
