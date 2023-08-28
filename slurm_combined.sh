#!/bin/bash
#### entire workflow for one variable, overwrites jobname and log files in sanity_checks.sh and merge_files.sh ####

tile=$1
var=$2
outfile_sanity=/p/tmp/annabu/projects/attrici/log/attrici_03_era5_t${tile}_${var}_rechunked/sanity_check_${tile}_${var}.log
outfile_merge=/p/tmp/annabu/projects/attrici/log/attrici_03_era5_t${tile}_${var}_rechunked/merge_files_${tile}_${var}.log


# create trace and ts files
cd /p/tmp/annabu/projects/attrici/runscripts/attrici_automated_processing/${tile}/attrici_03_era5_t${tile}_${var}_rechunked/
jobid=$(sbatch --parsable slurm.sh)

# sanity check
cd /p/tmp/annabu/projects/attrici/
jobid_sanity=$(sbatch --parsable --dependency=afterany:$jobid --job-name=attrici_sanity_checks_${tile}_${var} --output=${outfile_sanity} --error=${outfile_sanity} sanity_checks.sh ${tile} ${var} )

# merge traces
jobid_merged=$(sbatch --parsable --dependency=afterok:$jobid_sanity --job-name=attrici_merge_files_${tile}_${var} --output=${outfile_merge} --error=${outfile_merge} merge_files.sh ${tile} ${var})  

# create final cfacts
cd /p/tmp/annabu/projects/attrici/runscripts/attrici_automated_processing/${tile}/attrici_03_era5_t${tile}_${var}_rechunked/
sbatch --dependency=afterok:$jobid_sanity submit_write_netcdf.sh   #  create netcdf

