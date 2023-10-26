#!/bin/bash

rundir=runscripts/create_ssa_smoothed_gmt
mkdir $rundir 

cp settings.py $rundir
cp preprocessing/calc_gmt_by_ssa.py $rundir
cp preprocessing/create_ssa_smoothed_gmt.sh $rundir
cp ./variables_for_shellscripts.sh $rundir
cp preprocessing/creat_gmt_ssa_wrapper.sh $rundir

