#!/bin/bash

#SBATCH --partition=priority
#SBATCH --qos=priority
#SBATCH --job-name=merge_ERA5
#SBATCH --output=/p/tmp/sitreu/log/attrici/create_era5_gmt/log_merge_%j.log
#SBATCH --error=/p/tmp/sitreu/log/attrici/create_era5_gmt/log_merge_%j.log

TMP_DIR="/p/tmp/sitreu/data/tmp"
OUTPUT_DIR="/p/projects/ou/rd3/dmcci/basd_era5-land_to_efas-meteo/attrici_input/ERA5"
OUTPUT_FILES=""

for YEAR in {1950..2020}; do
    if [ "$YEAR" -eq 2020 ]; then
        OUTPUT_FILES+="${TMP_DIR}/gmt_daily_ERA5_${YEAR}.nc"
    else
        OUTPUT_FILES+="${TMP_DIR}/gmt_daily_ERA5_${YEAR}.nc "
    fi
done

cdo mergetime $OUTPUT_FILES "${OUTPUT_DIR}/merged_global_daily_means.nc"
# echo "cdo mergetime $OUTPUT_FILES ${OUTPUT_DIR}/merged_global_daily_means.nc"

# rm OUTPUT_FILES

