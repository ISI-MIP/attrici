#!/bin/bash
#### run script via bash command which implements sbatch job   ##

#SBATCH --qos=short # for smaller tiles
#SBATCH --partition=standard   # for smaller tiles
#SBATCH --job-name=attrici_merge
#SBATCH --account=dmcci
#SBATCH --output=/p/tmp/annabu/projects/attrici/log/create_backups/merge_files_%A.log
#SBATCH --error=/p/tmp/annabu/projects/attrici/log/create_backups/merge_files_%A.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=annabu@pik-potsdam.de
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
##SBATCH --time=00-02:00:00
#SBATCH --time=07:00:00


tile=$1
var=$2


# merge trace and ts files
/home/annabu/.conda/envs/attrici_pymc5_2/bin/python -u ./sanity_check/merge_files.py ${tile} ${var} || exit 1

echo "Finished, created backup files for " ${tile} ${var} 


EOT