#!/usr/bin/env bash
set -euo pipefail

# Install Lean/Lake tooling for CI and local verification.
#
# Usage:
#   ./scripts/setup-lean-ci-deps.sh
#   ./scripts/setup-lean-ci-deps.sh --check
#
# Optional environment variables:
#   ELAN_INIT_URL=https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh
#   LEAN_TOOLCHAIN=stable
#   INSTALL_RETRIES=2
#   HTTP_PROXY / HTTPS_PROXY / ALL_PROXY

CHECK_ONLY=0
if [[ "${1:-}" == "--check" ]]; then
  CHECK_ONLY=1
fi

INSTALL_RETRIES="${INSTALL_RETRIES:-2}"
ELAN_INIT_URL="${ELAN_INIT_URL:-https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh}"
LEAN_TOOLCHAIN="${LEAN_TOOLCHAIN:-stable}"

have_cmd() { command -v "$1" >/dev/null 2>&1; }

retry() {
  local attempt=1
  local max=$((INSTALL_RETRIES + 1))
  until "$@"; do
    if (( attempt >= max )); then
      echo "[retry] command failed after $attempt attempt(s): $*"
      return 1
    fi
    attempt=$((attempt + 1))
    echo "[retry] retrying ($attempt/$max): $*"
    sleep 1
  done
}

print_state() {
  echo "[state] elan: $(command -v elan || echo missing)"
  echo "[state] lean: $(command -v lean || echo missing)"
  echo "[state] lake: $(command -v lake || echo missing)"
}

if [[ "$CHECK_ONLY" -eq 1 ]]; then
  print_state
  if have_cmd lean && have_cmd lake; then
    exit 0
  fi
  exit 1
fi

if have_cmd lean && have_cmd lake; then
  echo "[setup] lean and lake already installed"
  print_state
  exit 0
fi

if ! have_cmd curl && ! have_cmd wget; then
  echo "[setup] missing curl/wget; cannot bootstrap elan"
  print_state
  exit 2
fi

tmpfile="$(mktemp)"
cleanup() { rm -f "$tmpfile"; }
trap cleanup EXIT

echo "[setup] downloading elan installer from: $ELAN_INIT_URL"
if have_cmd curl; then
  retry curl -fsSL "$ELAN_INIT_URL" -o "$tmpfile"
else
  retry wget -qO "$tmpfile" "$ELAN_INIT_URL"
fi

echo "[setup] running elan installer"
sh "$tmpfile" -y --default-toolchain "$LEAN_TOOLCHAIN"

# shellcheck disable=SC1091
if [[ -f "$HOME/.elan/env" ]]; then
  source "$HOME/.elan/env"
else
  export PATH="$HOME/.elan/bin:$PATH"
fi

if ! have_cmd lean || ! have_cmd lake; then
  echo "[setup] elan installed but lean/lake not found on PATH"
  print_state
  exit 3
fi

echo "[setup] lean toolchain details"
lean --version
lake --version
print_state
