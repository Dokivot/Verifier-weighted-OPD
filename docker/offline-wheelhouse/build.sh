#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace
OUTPUT=/out
WHEELS="$OUTPUT/wheels"
REQUIREMENTS="$OUTPUT/requirements-linux-cp311.txt"
PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://pypi.org/simple}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

mkdir -p "$WHEELS"
cd "$ROOT"

uv export --frozen \
  --extra data \
  --extra gpu \
  --extra eval \
  --extra tracking \
  --extra dev \
  --no-emit-project \
  --format requirements.txt \
  --no-annotate \
  --output-file "$REQUIREMENTS"

python -m pip download \
  --requirement "$REQUIREMENTS" \
  --dest "$WHEELS" \
  --only-binary=:all: \
  --index-url "$PYPI_INDEX_URL" \
  --extra-index-url "$PYTORCH_INDEX_URL"

python -m pip download \
  --dest "$WHEELS" \
  --only-binary=:all: \
  --index-url "$PYPI_INDEX_URL" \
  "uv" \
  "setuptools>=75" \
  "wheel"

python -m pip install \
  --dry-run \
  --ignore-installed \
  --no-index \
  --find-links "$WHEELS" \
  --requirement "$REQUIREMENTS" \
  >/tmp/opd-wheelhouse-offline-check.txt

cat >"$OUTPUT/manifest.txt" <<EOF
python=$(python --version 2>&1)
platform=$(python -c 'import platform; print(platform.platform())')
machine=$(python -c 'import platform; print(platform.machine())')
wheel_count=$(find "$WHEELS" -maxdepth 1 -type f -name '*.whl' | wc -l | tr -d ' ')
requirements=$REQUIREMENTS
EOF

cat "$OUTPUT/manifest.txt"
echo "Offline wheelhouse written to: $WHEELS"
