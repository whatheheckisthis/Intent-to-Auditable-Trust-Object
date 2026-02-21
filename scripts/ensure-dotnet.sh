#!/usr/bin/env bash
# CI guard entrypoint for deterministic offline dotnet recovery.
set -euo pipefail

EXIT_NO_ARCHIVE=42
SEGMENT_MS="${TME_SEGMENT_ENSURE_DOTNET_MS:-1800}"

# shellcheck disable=SC1091
source "$(dirname "$0")/timing_mitigation_engine.sh"

log() { printf '[ensure-dotnet] %s\n' "$*"; }
fail() { printf '[ensure-dotnet][error] %s\n' "$*" >&2; exit 1; }

ensure_dotnet() {
  if command -v dotnet >/dev/null 2>&1; then
    if dotnet --info >/dev/null 2>&1; then
      log "dotnet already available: $(dotnet --version)"
      return 0
    fi
    log "dotnet is present on PATH but unusable; invoking offline recovery"
  else
    log "dotnet not found on PATH; invoking offline recovery"
  fi

  set +e
  "$(dirname "$0")/recover-dotnet-from-archive.sh"
  local rc=$?
  set -e

  if [[ ${rc} -eq 0 ]]; then
    # shellcheck disable=SC1091
    source "$(dirname "$0")/activate-dotnet-offline.sh"
    log "dotnet ready: $(dotnet --version)"
    return 0
  fi

  if [[ ${rc} -eq ${EXIT_NO_ARCHIVE} ]]; then
    printf '[ensure-dotnet][error] offline recovery failed: no staged SDK archive available.\n' >&2
    printf '[ensure-dotnet][error] searched: /opt/bootstrap, %s/bootstrap, /workspace/bootstrap\n' "$HOME" >&2
    return ${EXIT_NO_ARCHIVE}
  fi

  printf '[ensure-dotnet][error] offline recovery failed with exit code %s\n' "${rc}" >&2
  return "${rc}"
}

tme_run_segment "ensure-dotnet" "${SEGMENT_MS}" -- ensure_dotnet "$@"
