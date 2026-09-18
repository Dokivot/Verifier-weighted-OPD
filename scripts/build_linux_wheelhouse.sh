#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUTPUT="${1:-$ROOT/opd-wheelhouse}"
IMAGE="${OPD_WHEELHOUSE_IMAGE:-opd-linux-wheelhouse:py311}"
PLATFORM="${DOCKER_DEFAULT_PLATFORM:-linux/amd64}"

if ! command -v docker >/dev/null; then
  echo "Docker is required. Install Docker Desktop and retry." >&2
  exit 2
fi

mkdir -p "$OUTPUT"
docker build \
  --platform "$PLATFORM" \
  -f docker/offline-wheelhouse/Dockerfile \
  -t "$IMAGE" \
  .

docker run --rm \
  --platform "$PLATFORM" \
  -e PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://pypi.org/simple}" \
  -e PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}" \
  -v "$ROOT:/workspace:ro" \
  -v "$OUTPUT:/out" \
  "$IMAGE"

echo "Wheelhouse ready: $OUTPUT/wheels"
echo "Upload this directory together with requirements-linux-cp311.txt and manifest.txt."
