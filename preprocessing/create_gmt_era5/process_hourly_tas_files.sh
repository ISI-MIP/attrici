#!/bin/bash

YEAR=$1
INPUT_DIR="/p/projects/climate_data_central/reanalysis/ERA5/tas"
TMP_DIR="/p/tmp/sitreu/data/tmp"

INPUT_FILE="${INPUT_DIR}/tas_1hr_ECMWF-ERA5_observation_${YEAR}010100-${YEAR}123123.nc"
TMP_OUTPUT_FILE="${TMP_DIR}/gmt_daily_ERA5_${YEAR}.nc"

# Compute daily and spatial mean using piping
cdo -L -O fldmean -daymean $INPUT_FILE $TMP_OUTPUT_FILE

