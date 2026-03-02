#!/usr/bin/env bash
set -euo pipefail

# Normalize proxy-related environment variables for CI tools and containers.
# - Mirrors lowercase/uppercase proxy env names
# - Ensures NO_PROXY includes local loopback defaults
# - Exports values for subsequent GitHub Actions steps via GITHUB_ENV

first_set() {
  local v
  for v in "$@"; do
    if [[ -n "${!v:-}" ]]; then
      printf '%s' "${!v}"
      return 0
    fi
  done
  return 1
}

HTTP_PROXY_VAL="$(first_set HTTP_PROXY http_proxy || true)"
HTTPS_PROXY_VAL="$(first_set HTTPS_PROXY https_proxy || true)"
NO_PROXY_VAL="$(first_set NO_PROXY no_proxy || true)"

DEFAULT_NO_PROXY="localhost,127.0.0.1,::1"
if [[ -z "$NO_PROXY_VAL" ]]; then
  NO_PROXY_VAL="$DEFAULT_NO_PROXY"
elif [[ ",$NO_PROXY_VAL," != *",localhost,"* ]]; then
  NO_PROXY_VAL="$NO_PROXY_VAL,$DEFAULT_NO_PROXY"
fi

export HTTP_PROXY="$HTTP_PROXY_VAL"
export http_proxy="$HTTP_PROXY_VAL"
export HTTPS_PROXY="$HTTPS_PROXY_VAL"
export https_proxy="$HTTPS_PROXY_VAL"
export NO_PROXY="$NO_PROXY_VAL"
export no_proxy="$NO_PROXY_VAL"

if [[ -n "${GITHUB_ENV:-}" ]]; then
  {
    echo "HTTP_PROXY=$HTTP_PROXY"
    echo "http_proxy=$http_proxy"
    echo "HTTPS_PROXY=$HTTPS_PROXY"
    echo "https_proxy=$https_proxy"
    echo "NO_PROXY=$NO_PROXY"
    echo "no_proxy=$no_proxy"
  } >> "$GITHUB_ENV"
fi

echo "[ci-env] HTTP_PROXY set: $([[ -n "$HTTP_PROXY" ]] && echo yes || echo no)"
echo "[ci-env] HTTPS_PROXY set: $([[ -n "$HTTPS_PROXY" ]] && echo yes || echo no)"
echo "[ci-env] NO_PROXY=$NO_PROXY"
