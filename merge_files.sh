#!/bin/bash
#### merge files after passed sanity checks  ####

tile=$1
trace_or_ts=$2
 

# merge trace and ts files for each variable after they passed sanity checks
for var in tas0 tas6 tas12 tasrange pr0 pr6 pr12 pr18 sfcWind rsds hurs; do  
#for var in tas0 tas6 tas12 tas18 tasrange tasskew pr0 pr6 pr12 pr18 sfcWind rsds hurs; do  
  echo "Merging: " ${var}
  /home/annabu/.conda/envs/attrici_pymc5_2/bin/python -u sanity_check/merge_files.py ${tile} ${trace_or_ts} ${var}
done

echo "Finished, merged all" ${trace_or_ts}