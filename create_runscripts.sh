#!/bin/bash
source ./variables_for_shellscripts.sh
tile=$1
function copy_files_and_configure() {
  rundir=attrici_03_era5_t${tile}_${var}${hour}_rechunked
  mkdir $rundir
  cp ../../../.pytensorrc $rundir/
  cp ../../../run_estimation.py $rundir/
  cp ../../../slurm.sh $rundir/
  cp ../../../settings.py $rundir/
  cp ../../../write_netcdf.py $rundir/
  cp ../../../submit_write_netcdf.sh $rundir/
  cp ../../../sanity_check.py $rundir/
  cp ../../../sanity_checks.sh $rundir/
  cp ../../../visual_check.py $rundir/
  cp ../../../visual_checks.sh $rundir/
  cp ../../../variables_for_shellscripts $rundir/

  sed -i -e 's/hour = ""/hour = "'"${hour}"'"/' $rundir/settings.py 
  sed -i -e 's/variable = "tas"/variable = "'"${var}"'"/' $rundir/settings.py 
  sed -i -e 's/tile = "00009"/tile = "'"${tile}"'"/' $rundir/settings.py 
  sed -i -e 's/--job-name=runid_merge/--job-name='$rundir'/' $rundir/submit_write_netcdf.sh
  sed -i -e 's/--job-name=attrici_run_estimation/--job-name='$rundir'/' $rundir/slurm.sh
  mkdir $project_basedir/log/$rundir
}

mkdir runscripts/attrici_automated_processing/${tile}
cd runscripts/attrici_automated_processing/${tile}

for var in tas tasrange tasskew sfcWind rsds hurs;
do
  hour=""
  copy_files_and_configure
done
for var in tas pr
# for var in pr;
do
  for hour in 0 6 12 18;
  do
    copy_files_and_configure
  done
done
