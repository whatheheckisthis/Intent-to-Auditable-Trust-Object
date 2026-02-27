#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

log() { echo "[lint] $*"; }

if command -v flake8 >/dev/null 2>&1; then
  log "running flake8 against src tests ci/tools"
  flake8 "$REPO_ROOT/src" "$REPO_ROOT/tests" "$REPO_ROOT/ci/tools"
  log "flake8 completed successfully"
  exit 0
fi

log "flake8 not found; running syntax-only fallback"
python3 -m compileall -q "$REPO_ROOT/src" "$REPO_ROOT/tests" "$REPO_ROOT/ci/tools"
log "syntax-only fallback completed successfully"
