# ERA5 Hourly Temperature Data Processing Pipeline

This pipeline processes the hourly temperature data from the ERA5 dataset for the years 1950-2020. The goal is to compute the daily and spatial mean temperatures and merge the processed data into a single file.

## Steps:

1. **Hourly to Daily and Spatial Mean**:
    - The script `process_hourly_tas_files.sh` processes each file to compute the daily mean followed by the spatial mean. 
    - The SLURM script `slurm_process_hourly_tas.sh` distributes the jobs using job arrays for the specified years.

2. **Merging Processed Data**:
    - Once all SLURM jobs are completed successfully, the script `merge_gmt_daily_files.sh` merges the processed files into a single file. If any jobs fail or get canceled, a notification is triggered.

3. **Wrapper Script**:
    - The `wrapper_script.sh` takes care of the overall execution flow.

## Usage:
Execute the wrapper script to run the entire pipeline:

