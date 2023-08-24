#!/bin/bash
#### entire workflow for one variable, overwrites jobname and log files in sanity_checks.sh and merge_files.sh ####

#SBATCH --qos=priority
#SBATCH --partition=priority
#SBATCH --job-name=attrici_jobs_combi
#SBATCH --account=dmcci
#SBATCH --output=/p/tmp/annabu/projects/attrici/log/%A_combi.log
#SBATCH --error=/p/tmp/annabu/projects/attrici/log/%A_combi.log
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=annabu@pik-potsdam.de
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1  
#SBATCH --time=02:00

tile=$1
var=$2
outfile=/p/tmp/annabu/projects/attrici/log/attrici_03_era5_t${tile}_${var}_rechunked/sanity_check_merge_files_${tile}_${var}.log


# create trace and ts files
cd /p/tmp/annabu/projects/attrici/runscripts/attrici_automated_processing/${tile}/attrici_03_era5_t${tile}_${var}_rechunked/
jobid=$(sbatch --parsable slurm.sh)

# sanity check and # merge traces
cd /p/tmp/annabu/projects/attrici/
jobid_sanity=$(sbatch --parsable --job-name=attrici_sanity_checks_${tile}_${var} --output=${outfile} --error=${outfile} sanity_checks.sh ${tile} ${var} )
jobid_merged=$(sbatch --parsable --dependency=afterok:${jobid_sanity}:+3 --job-name=attrici_merge_files_${tile}_${var} --output=${outfile} --error=${outfile} merge_files.sh ${tile} ${var})  

# create final cfacts
cd /p/tmp/annabu/projects/attrici/runscripts/attrici_automated_processing/${tile}/attrici_03_era5_t${tile}_${var}_rechunked/
#sbatch --dependency=afterok:${jobid_sanity}:+3 submit_write_netcdf.sh   #  create netcdf
echo "Skipping job to merge timeseries to final cfacts" 
