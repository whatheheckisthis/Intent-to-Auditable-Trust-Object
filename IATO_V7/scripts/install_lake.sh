#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$SCRIPT_DIR/env.sh"

configure_env() {
  # Keep Lean tooling predictable for local shells and CI.
  export ELAN_HOME="${ELAN_HOME:-$HOME/.elan}"
  export PATH="$ELAN_HOME/bin:$PATH"
  export LAKE="$ELAN_HOME/bin/lake"

  # Stable defaults for Lean/Lake build behavior.
  export LEAN_ABORT_ON_PANIC="${LEAN_ABORT_ON_PANIC:-1}"
  export LEAN_PATH="${LEAN_PATH:-$PROJECT_ROOT/.lake/packages}"
  export LAKE_NO_CACHE="${LAKE_NO_CACHE:-0}"
}

write_env_file() {
  cat > "$ENV_FILE" <<ENV
# shellcheck shell=bash
# Source this file before running lake commands:
#   source scripts/env.sh

export ELAN_HOME="${ELAN_HOME:-$HOME/.elan}"
export PATH="\$ELAN_HOME/bin:\$PATH"
export LAKE="\$ELAN_HOME/bin/lake"

# Lean/Lake build defaults
export LEAN_ABORT_ON_PANIC="${LEAN_ABORT_ON_PANIC:-1}"
export LEAN_PATH="${LEAN_PATH:-$PROJECT_ROOT/.lake/packages}"
export LAKE_NO_CACHE="${LAKE_NO_CACHE:-0}"
ENV
}

configure_env

if command -v lake >/dev/null 2>&1; then
  echo "lake already installed: $(command -v lake)"
  write_env_file
  lake --version || true
  echo "Environment helper written: $ENV_FILE"
  exit 0
fi

echo "[1/4] Checking for elan..."
if ! command -v elan >/dev/null 2>&1; then
  echo "elan not found. Attempting install from GitHub release..."
  tmpdir="$(mktemp -d)"
  trap 'rm -rf "$tmpdir"' EXIT

  if curl -fL "https://github.com/leanprover/elan/releases/latest/download/elan-x86_64-unknown-linux-gnu.tar.gz" -o "$tmpdir/elan.tar.gz"; then
    tar -xzf "$tmpdir/elan.tar.gz" -C "$tmpdir"
    "$tmpdir/elan-init" -y
  else
    echo "GitHub download failed. Trying apt fallback (requires sudo/root and network access)..."
    if command -v apt-get >/dev/null 2>&1; then
      apt-get update
      apt-get install -y elan
    else
      echo "apt-get unavailable; cannot auto-install elan."
      exit 1
    fi
  fi
fi

configure_env

if ! command -v elan >/dev/null 2>&1; then
  echo "elan installation did not complete successfully."
  exit 1
fi

echo "[2/4] Installing Lean stable toolchain..."
elan default stable

echo "[3/4] Verifying lean + lake binaries..."
command -v lean
command -v lake

lean --version
lake --version

echo "[4/4] Writing environment helper..."
write_env_file

echo "Done. Next steps:"
echo "  source scripts/env.sh"
echo "  lake update && lake build && lake test"
