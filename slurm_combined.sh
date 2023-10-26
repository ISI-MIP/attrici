#!/bin/bash
#### entire workflow for one variable, overwrites jobname and log files in sanity_checks.sh and merge_files.sh ####
source ./variables_for_shellscripts.sh

tile=$1
var=$2

logdir=$project_basedir/log/attrici_04_era5_t${tile}_${var}_rechunked
mkdir -p $logdir
logdir_slurm=$logdir/run_estimation
mkdir -p $logdir_slurm
logfile_slurm=$logdir_slurm/%x_%j
logfile=$logdir/%x_%j 
user=$(whoami)
# create trace and ts files
cd $project_basedir/runscripts/attrici_automated_processing/${tile}/attrici_04_era5_t${tile}_${var}_rechunked/
jobid=$(sbatch --parsable --mail-user=$user@pik-potsdam.de --output=$logfile_slurm --error=$logfile_slurm slurm.sh)

# sanity check
jobid_sanity=$(sbatch --parsable --mail-user=$user@pik-potsdam.de --dependency=afterany:$jobid --job-name=attrici_sanity_checks_${tile}_${var} --output=$logfile --error=${logfile} sanity_checks.sh)

# create final cfacts
jobid_cfact=$(sbatch --parsable --mail-user=$user@pik-potsdam.de --dependency=afterok:$jobid_sanity --job-name=merge_timeseries_${tile}_${var} --output=${logfile} --error=${logfile} submit_write_netcdf.sh)   #  create netcdf

# visual check of created cfacts
jobid_visualcheck=$(sbatch --parsable --dependency=afterok:$jobid_cfact --mail-user=$user@pik-potsdam.de --job-name=attrici_visual_checks_${tile}_${var} --output=${logfile} --error=${logfile} visual_checks.sh)

# merge traces
# sbatch --dependency=afterok:$jobid_sanity --mail-user=$user@pik-potsdam.de --job-name=attrici_merge_traces_${tile}_${var} --output=${outfile_merge} --error=${outfile_merge} merge_files.sh ${tile} ${var}

## move final cfact and backup file to project folder, delete source files
# cd $project_basedir/output/
# sbatch --dependency=afterok:$jobid_visualcheck --mail-user=$user@pik-potsdam.de --job-name=attrici_move_final_cfact_${tile}_${var} --error=${outfile_visualcheck} move_final_cfacts.sh ${tile} ${var}

