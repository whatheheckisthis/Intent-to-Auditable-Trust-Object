#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$SCRIPT_DIR/verify_environment.sh"
python3 "$SCRIPT_DIR/verify_imports.py"

if command -v flake8 >/dev/null 2>&1; then
  bash "$SCRIPT_DIR/lint.sh"
else
  echo "WARN: flake8 not available; skipping lint step."
fi

echo "All architecture CLI checks passed."
