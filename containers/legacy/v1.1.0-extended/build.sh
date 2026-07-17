#!/usr/bin/env bash
# Build v1.1.0-extended Singularity image on the cluster (requires rootless singularity/apptainer).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMMIT="${1:-a83e74f}"
OUT="${2:-/p/projects/isimip/isimip/sitreu/containers/attrici-v1.1.0-extended-${COMMIT}.sif}"
DEF="${REPO_ROOT}/containers/legacy/v1.1.0-extended/attrici-v1.1.0-extended.def"

echo "Building ${OUT} from ${DEF}"
echo "Ensure legacy/v1.1.0-extended is pushed to GitHub at commit ${COMMIT} before building."
singularity build "${OUT}" "${DEF}"
echo "Built: ${OUT}"
