#!/bin/bash

# Submit the SLURM job
JOB_ID=$(sbatch --parsable slurm_process_hourly_tas.sh)

# Submit the merging job with dependency on successful completion
after_ok_job_id=$(sbatch --parsable --depend=afterok:$JOB_ID merge_gmt_daily_files.sh)

# Submit the notification job if any of the array jobs fail or get canceled
sbatch --depend=afternotok:$JOB_ID notify_failure.sh $JOB_ID $after_ok_job_id

