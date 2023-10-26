#!/bin/bash

#SBATCH --partition=priority
#SBATCH --qos=priority
#SBATCH --job-name=notification
#SBATCH --output=/p/tmp/sitreu/log/attrici/create_era5_gmt/log_notify_%j.log
#SBATCH --error=/p/tmp/sitreu/log/attrici/create_era5_gmt/log_notify_%j.log
#SBATCH --account=dmcci
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sitreu@pik-potsdam.de


failed_job_id=$1
after_ok_job_id=$2

echo "One or more jobs from the array $failed_job_id failed or were canceled!"

TMP_DIR="/p/tmp/sitreu/data/tmp"
OUTPUT_DIR="/p/projects/ou/rd3/dmcci/basd_era5-land_to_efas-meteo/attrici_input/ERA5"
OUTPUT_FILES=""

for YEAR in {1950..2020}; do
    OUTPUT_FILES+="${TMP_DIR}/gmt_daily_ERA5_${YEAR}.nc "
done

echo "remove output files: $OUTPUT_FILES"

rm OUTPUT_FILES


echo "canceling dependency: job_id $after_ok_job_id"

scancel $after_ok_job_id

# You can also add mail commands or other notifications here if needed.

