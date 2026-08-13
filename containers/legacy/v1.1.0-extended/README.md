# Legacy Singularity container for attrici v1.1.0-extended

Singularity/Apptainer image that runs `run_estimation.py` from attrici branch **legacy/v1.1.0-extended** (Python 3.7, PyMC3, Theano).

`ATTRICI_COMMIT` in the `.def` is the pin. `build.sh` names the `.sif` after that SHA and checks `attrici.__version__` inside the image.

Push `legacy/v1.1.0-extended` to GitHub before building.

## Build

Requires **root** (`sudo`) and network access.

1. Set `ATTRICI_COMMIT` in `attrici-v1.1.0-extended.def` and commit it.
2. Build:

```bash
cd /path/to/attrici
sudo bash containers/legacy/v1.1.0-extended/build.sh
```

This writes `attrici-v1.1.0-extended-<sha>.sif` (currently `…-8d7e1c3.sif`).

Copy to cluster:

```bash
scp containers/legacy/v1.1.0-extended/attrici-v1.1.0-extended-8d7e1c3.sif \
  login:/p/projects/isimip/isimip/sitreu/containers/
```

Point attrici-workflow `singularity_image` and `resolved_commit` at the same SHA.

## Run

```bash
singularity run -B /path/to/your/workflow:/workspace attrici-v1.1.0-extended-8d7e1c3.sif
```
