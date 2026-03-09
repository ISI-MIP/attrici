# Legacy Singularity container for attrici v1.1.0

Singularity image that runs `run_estimation.py` from attrici tag **v1.1.0** (Python 3.7, PyMC3, Theano). The recipe clones the repo at v1.1.0 inside the container at build time; no need to checkout v1.1.0 on the host.

## Build

From the repository root (requires network so Singularity can pull the Docker base image and clone the repo):

```bash
singularity build attrici-v1.1.0.sif containers/legacy/v1.1.0/attrici-v1.1.0.def
```

Or from this directory:

```bash
singularity build attrici-v1.1.0.sif attrici-v1.1.0.def
```

## Run

Default run executes `run_estimation.py` (which reads `settings` and expects input/output paths and optional Slurm env vars):

```bash
singularity run attrici-v1.1.0.sif
```

### Use from a workflow (e.g. Snakemake)

The container is set up so that **the workflow directory can live anywhere on the host**. It does not need to know the host path at build time.

- **Fixed mount point in    side the container:** `/workspace`
- **Default WORKDIR:** `/workspace`

Bind-mount your workflow’s experiment directory (e.g. `attrici-workflow/runscripts/exp1`) to `/workspace`. The runscript will use that directory as the current working directory and, if `run_estimation.py` exists there, run it (so your `settings.py` in the same directory is used). If `run_estimation.py` is not present in the mounted dir, the container runs its built-in `/opt/attrici/run_estimation.py`.

Example (host path can be anything):

```bash
# Workflow lives at /home/user/attrici-workflow or /p/project/attrici-workflow – same invocation
singularity run -B /path/to/attrici-workflow/runscripts/experiment1:/workspace attrici-v1.1.0.sif
```

No need to set `WORKDIR` unless you use a different mount path; then set `WORKDIR` to that path (e.g. `--env WORKDIR=/workspace`).

To use the container’s built-in script with custom input/output dirs only:

```bash
singularity run -B /path/to/input:/data/input -B /path/to/output:/data/output attrici-v1.1.0.sif
```

**Theano config:** The container uses the v1.1.0 repo’s `config/theanorc` via `THEANORC=/opt/attrici/config/theanorc`. That file is included from the cloned tag. If you don’t set `THEANORC`, Theano would otherwise use `$HOME/.theanorc`; because Singularity binds your host `$HOME` by default, the host’s `~/.theanorc` would be used. By setting `THEANORC` in the image, the repo’s theanorc is used by default. Override with `--env THEANORC=/path/to/your/theanorc` if needed (e.g. the v1.1.0 file has host-style paths for cuda/blas that you may want to replace on your system).

For job arrays (e.g. Slurm), set the same variables the original v1.1.0 submit script used:

- `SUBMITTED=1`
- `SLURM_ARRAY_TASK_ID`, `SLURM_ARRAY_TASK_COUNT`
- `THEANO_FLAGS` (e.g. `base_compiledir=/tmp/...`) if you want a dedicated Theano compile dir

Example with env and binds:

```bash
export SUBMITTED=1
export THEANO_FLAGS="base_compiledir=/tmp/theano_$$"
singularity run -B /path/to/input:/data/input -B /path/to/output:/data/output attrici-v1.1.0.sif
```

## Contents

- **Base:** `continuumio/miniconda3:4.9.2` (Bootstrap: docker)
- **Compiler:** g++-7 (for Theano)
- **Conda env:** `attrici` with Python 3.7, PyMC3, Theano, netCDF4, pandas, xarray, pytables, etc.
- **Code:** attrici at tag v1.1.0 in `/opt/attrici`
- **Default workdir:** `/workspace` (fixed mount point for workflows; set `WORKDIR` to override)
