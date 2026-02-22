#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WHEEL_DIR="${ROOT_DIR}/build/guest/wheels"
mkdir -p "${WHEEL_DIR}"

python3 -m pip download \
  --platform manylinux_2_28_aarch64 \
  --python-version 311 \
  --only-binary :all: \
  --dest "${WHEEL_DIR}" \
  cryptography==42.0.5 pytest==8.1.0 pytest-cov==5.0.0 cffi tpm2-pytss

count="$(find "${WHEEL_DIR}" -maxdepth 1 -type f -name '*.whl' | wc -l | tr -d ' ')"
echo "[fetch-wheels] OK: ${count} wheels downloaded to build/guest/wheels/"
