#!/usr/bin/env bash
set -euo pipefail

if ! command -v flake8 >/dev/null 2>&1; then
  echo "ERROR: flake8 is not installed"
  exit 1
fi

flake8 .
