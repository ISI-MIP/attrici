#!/bin/bash
#### sanity slurm, run shell script via bash command, which starts job  ###

#SBATCH --qos=priority
#SBATCH --partition=priority
#SBATCH --job-name=attrici_sanity_checks
#SBATCH --account=dmcci
#SBATCH --output=/p/tmp/annabu/projects/attrici/log/sanity_checks/sanity_check_%A.log
#SBATCH --error=/p/tmp/annabu/projects/attrici/log/sanity_checks/sanity_check_%A.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=annabu@pik-potsdam.de
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --array=1  #test with 1 for combi_slurm.sh, test if more speeds up
#SBATCH --time=05:00


tile=$1
var=$2

#for var in tas0 tas6 tas12 tas18 pr0 pr6 pr12 pr18 tasrange tasskew hurs rsds sfcWind; 
#do 
echo "Checking: " ${tile} ${var}
/home/annabu/.conda/envs/attrici_pymc5_2/bin/python -u sanity_check/sanity_check.py ${tile} ${var}
exit 0
#done
echo "Finished checks"
