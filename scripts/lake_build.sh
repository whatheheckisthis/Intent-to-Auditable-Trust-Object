#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" ]]; then
  cat <<'USAGE'
Usage: scripts/lake_build.sh
Builds Lean targets from the repository's lean/ project.
USAGE
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LEAN_ROOT="$REPO_ROOT/lean"

if [[ ! -f "$LEAN_ROOT/lakefile.lean" ]]; then
  echo "error: expected Lean project at $LEAN_ROOT (missing lakefile.lean)" >&2
  exit 1
fi

cd "$LEAN_ROOT"
echo "Building IATO-V7 formal verification..."
lake update
lake build
lake build tests
echo "Build complete"
