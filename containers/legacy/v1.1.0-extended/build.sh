#!/usr/bin/env bash
# Build the image named after the attrici commit pinned in the .def.
#
#   1. Set ATTRICI_COMMIT in attrici-v1.1.0-extended.def and commit it
#   2. sudo bash containers/legacy/v1.1.0-extended/build.sh
#
# Writes attrici-v1.1.0-extended-<7-char-sha>.sif and checks attrici.__version__.
#
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEF="${DIR}/attrici-v1.1.0-extended.def"

SHA="$(sed -n 's/^ATTRICI_COMMIT="\([0-9a-fA-F]\{7,40\}\)".*/\1/p' "$DEF")"
if [[ -z "$SHA" ]]; then
  echo "ERROR: ${DEF} must set ATTRICI_COMMIT to a git SHA" >&2
  exit 1
fi
SHORT="${SHA:0:7}"
OUT="${DIR}/attrici-v1.1.0-extended-${SHORT}.sif"

echo "Building ${OUT} from ${DEF} (commit ${SHA})"
singularity build --force "$OUT" "$DEF"

version="$(singularity exec "$OUT" python -c "import attrici; print(attrici.__version__)")"
echo "Installed attrici ${version}"
if ! grep -qi "g${SHORT}" <<<"$version"; then
  echo "ERROR: expected attrici.__version__ to contain g${SHORT}" >&2
  exit 1
fi
ls -lh "$OUT"
