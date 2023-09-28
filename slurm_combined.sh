#!/bin/bash
#### entire workflow for one variable, overwrites jobname and log files in sanity_checks.sh and merge_files.sh ####

tile=$1
var=$2
outfile_sanity=/p/tmp/annabu/projects/attrici/log/attrici_03_era5_t${tile}_${var}_rechunked/sanity_check_${tile}_${var}.log
outfile_merge=/p/tmp/annabu/projects/attrici/log/attrici_03_era5_t${tile}_${var}_rechunked/merge_files_${tile}_${var}.log
outfile_visualcheck=/p/tmp/annabu/projects/attrici/log/attrici_03_era5_t${tile}_${var}_rechunked/final_cfact_visual_check_${tile}_${var}.log
user=$(whoami)


# create trace and ts files
cd /p/tmp/annabu/projects/attrici/runscripts/attrici_automated_processing/${tile}/attrici_03_era5_t${tile}_${var}_rechunked/
jobid=$(sbatch --parsable --mail-user=$user@pik-potsdam.de slurm.sh)

# sanity check
cd /p/tmp/annabu/projects/attrici/
jobid_sanity=$(sbatch --parsable --mail-user=$user@pik-potsdam.de --dependency=afterany:$jobid --job-name=attrici_sanity_checks_${tile}_${var} --output=${outfile_sanity} --error=${outfile_sanity} sanity_checks.sh ${tile} ${var})

# merge traces
sbatch --dependency=afterok:$jobid_sanity --mail-user=$user@pik-potsdam.de slurm.sh --job-name=attrici_merge_files_${tile}_${var} --output=${outfile_merge} --error=${outfile_merge} merge_files.sh ${tile} ${var}

# create final cfacts
cd /p/tmp/annabu/projects/attrici/runscripts/attrici_automated_processing/${tile}/attrici_03_era5_t${tile}_${var}_rechunked/
jobid_cfact=$(sbatch --parsable --mail-user=$user@pik-potsdam.de slurm.sh --dependency=afterok:$jobid_sanity submit_write_netcdf.sh)   #  create netcdf

# visual check of created cfacts
cd /p/tmp/annabu/projects/attrici/
jobid_visualcheck=$(sbatch --parsable --dependency=afterok:$jobid_cfact --mail-user=$user@pik-potsdam.de slurm.sh --job-name=attrici_visual_checks_${tile}_${var} --output=${outfile_visualcheck} --error=${outfile_visualcheck} visual_checks.sh ${tile} ${var})

## move final cfact and backup file to project folder, delete source files
cd /p/tmp/annabu/projects/attrici/output/
sbatch --dependency=afterok:$jobid_visualcheck --mail-user=$user@pik-potsdam.de slurm.sh --job-name=attrici_move_final_cfact_${tile}_${var} --error=${outfile_visualcheck} move_final_cfacts.sh ${tile} ${var}

