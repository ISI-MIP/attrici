# Legacy Singularity container for attrici v1.1.0-extended

Singularity image that runs `run_estimation.py` from attrici branch **legacy/v1.1.0-extended** (Python 3.7, PyMC3, Theano). Extends v1.1.0 with:

- `calibration_start` and `calibration_stop` configuration
- GMT normalization from calibration period only
- Time range alignment validation (GMT and input must cover same range)

The recipe clones the repo at legacy/v1.1.0-extended inside the container at build time. Push the branch to GitHub before building.

## Build

From the repository root (requires network):

```bash
singularity build attrici-v1.1.0-extended.sif containers/legacy/v1.1.0-extended/attrici-v1.1.0-extended.def
```

## Run

Same as v1.1.0. Bind-mount your workflow directory to `/workspace`:

```bash
singularity run -B /path/to/your/workflow:/workspace attrici-v1.1.0-extended.sif
```
