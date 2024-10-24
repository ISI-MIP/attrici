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
tune = 500
draws = 1000
chains = 2
```

This can be used to re-run a specific config

```
attrici detrend --config runconfig.toml
```
