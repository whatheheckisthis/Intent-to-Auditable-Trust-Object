#!/usr/bin/env bash
set -euo pipefail

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda is required for bootstrap"
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate testenv
conda install -y -c conda-forge numpy flake8 pytest

echo "Environment bootstrap complete for testenv."
