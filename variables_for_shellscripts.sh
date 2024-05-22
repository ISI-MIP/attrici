#! /bin/echo Please-source

# This file stores user specific paths for shell scripts in this project.

# attrici_python=$HOME/.conda/envs/attrici/bin/python
attrici_python_gmt=$HOME/.conda/envs/attrici_gmt/bin/python
project_basedir=/p/tmp/sitreu/projects/attrici
export RUNDIR=$(pwd)

attrici_python() {
  singularity exec -B /p:/p ${project_basedir}/ATTRICI.sif bash -c "cd $RUNDIR; python $@"
}

