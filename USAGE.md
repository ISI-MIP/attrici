# Usage

## Command line interface

Attrici makes its main functionality available via a command line tool.

The current version can be displayed with

```
attrici --version
```

Show the command line options

```
attrici --help
```

The attrici command line interface takes a sub-command to perform the speciefied operation

```
detrend             Detrend a dataset
merge-output        Merge detrended output
merge-traces        Merge traces from detrend run
preprocess-tas      Derive tasrange and tasskew from tas, tasmin, and tasmax
postprocess-tas     Derive tasmin and tasmax from tas, tasrange, and tasskew
ssa                 Perform singular spectrum analysis
```

For help on these sub-commands use e.g.

```
attrici detrend --help
```

### detrend

Some test data is provided in `tests/data` to demonstrate a `detrend` run.

```
attrici detrend --gmt-file ./tests/data/20CRv3-ERA5_germany_ssa_gmt.nc \
                --input-file ./tests/data/20CRv3-ERA5_germany_obs.nc \
                --variable tas \
                --stop-date "2023-12-31" \
                --report-variables ds y cfact logp
```

Some cells can be omitted from the calculation with a mask via the `--mask-file` option, e.g. a land-sea-mask.
The example below uses a mask to use only three grid cells.

```
attrici detrend --gmt-file ./tests/data/20CRv3-ERA5_germany_ssa_gmt.nc \
                --input-file ./tests/data/20CRv3-ERA5_germany_obs.nc \
                --mask-file ./tests/data/20CRv3-ERA5_germany_mask.nc \
                --variable tas \
                --stop-date "2023-12-31" \
                --report-variables ds y cfact logp
```

To change the logging level the `LOGURU_LEVEL` environment variable can be set.
For example, to show messages with level WARNING and higher

```
LOGURU_LEVEL=WARNING attrici detrend \
                --gmt-file ./tests/data/20CRv3-ERA5_germany_ssa_gmt.nc \
                --input-file ./tests/data/20CRv3-ERA5_germany_obs.nc \
                --variable tas \
                --stop-date "2023-12-31" \
                --report-variables ds y cfact logp
```

To print the current config set via defaults and command line options as TOML the `--print-config` flag can be used.

```
attrici detrend --gmt-file ./tests/data/20CRv3-ERA5_germany_ssa_gmt.nc \
                --input-file ./tests/data/20CRv3-ERA5_germany_obs.nc \
                --variable tas \
                --stop-date "2023-12-31" \
                --report-variables ds y cfact logp \
                --print-config
```

TOML output is shown below.

```
gmt_file = "tests/data/20CRv3-ERA5_germany_ssa_gmt.nc"
input_file = "tests/data/20CRv3-ERA5_germany_obs.nc"
variable = "tas"
output_dir = "."
overwrite = false
modes = [4, 4, 4, 4]
progressbar = false
report_variables = ["ds", "y", "cfact", "logp"]
seed = 0
stop_date = 2023-12-31
task_id = 0
task_count = 1
timeout = 3600
use_cache = false
```

This can be used to re-run a specific config

```
attrici detrend --config runconfig.toml
```

## Running on HPC platforms

As a computationally expensive operation, the `detrend` sub-command is designed to be run in parallel (for each geographical cell).
To make use of this parallelization, specify the arguments `--task-id ID` and `--task-count COUNT` and start several instances with `N` going from `0` to `N-1`. `N` does not have to equal the number of cells - these will be distributed to instances accordingly.

For the SLURM scheduler, which is widely used on HPC platforms, you can use an `sbatch` run script such as the following (here `N=4`):

```bash
#!/usr/bin/env bash
#SBATCH --account=MYACCOUNT
#SBATCH --array=0-3
#SBATCH --cpus-per-task=2
#SBATCH --export=ALL,OMP_PROC_BIND=TRUE
#SBATCH --job-name="attrici"
#SBATCH --ntasks=1
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --time=01:00:00

# load necessary modules/packages here if you don't queue with them loaded
# e.g.: module purge; module load ...
#   or: spack load ...

# load virtual environment if you don't queue with it activated:
# e.e.: source venv/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun attrici \
     detrend \
     --gmt-file ./tests/data/20CRv3-ERA5_germany_ssa_gmt.nc \
     --input-file ./tests/data/20CRv3-ERA5_germany_obs.nc \
     --output-dir ./tests/data/output \
     --variable tas \
     --stop-date 2021-12-31 \
     --report-variables ds y cfact logp \
     --overwrite \
     --task-id "$SLURM_ARRAY_TASK_ID" \
     --task-count "$SLURM_ARRAY_TASK_COUNT"
```

If you prefer SLURM tasks rather than job arrays, an example scheduling script would look like:

```bash
#!/usr/bin/env bash
#SBATCH --account=MYACCOUNT
#SBATCH --cpus-per-task=2
#SBATCH --export=ALL,OMP_PROC_BIND=TRUE
#SBATCH --job-name="attrici"
#SBATCH --ntasks=4
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --time=01:00:00

# load necessary modules/packages here if you don't queue with them loaded
# e.g.: module purge; module load ...
#   or: spack load ...

# load virtual environment if you don't queue with it activated:
# e.e.: source venv/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun bash <<'EOF'

exec attrici \
     detrend \
     --gmt-file ./tests/data/20CRv3-ERA5_germany_ssa_gmt.nc \
     --input-file ./tests/data/20CRv3-ERA5_germany_obs.nc \
     --output-dir ./tests/data/output \
     --variable tas \
     --stop-date 2021-12-31 \
     --report-variables ds y cfact logp \
     --overwrite \
     --task-id "$SLURM_PROCID" \
     --task-count "$SLURM_NTASKS"
EOF
```

Both scripts assume that you schedule them from a setup suitable to run ATTRICI, i.e. with a virtual environment activated being able to run ATTRICI locally.
Otherwise, adjust the scripts to setup that environment as given in the respective comment in the script.
