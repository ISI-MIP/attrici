#!/bin/bash

source ./variables_for_shellscripts.sh

logdir=$project_basedir/log/create_ssa_smoothed_gmt
mkdir -p $logdir
logfile=$logdir/%x_%j 
user=$(whoami)

sbatch --output=$logfile --error=${logfile} --mail-user=$user@pik-potsdam.de create_ssa_smoothed_gmt.sh
