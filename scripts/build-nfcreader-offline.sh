#!/usr/bin/env bash
# Build-staging workflow: compile-only validation, no restore/runtime execution.
set -euo pipefail

SEGMENT_MS="${TME_SEGMENT_BUILD_OFFLINE_MS:-8000}"

# shellcheck disable=SC1091
source "$(dirname "$0")/timing_mitigation_engine.sh"

log() { printf '[build-offline] %s\n' "$*"; }
fail() { printf '[build-offline][error] %s\n' "$*" >&2; exit 46; }

tme_run_segment "build-offline.ensure-dotnet" "${TME_SEGMENT_BUILD_ENSURE_MS:-2200}" -- bash "$(dirname "$0")/ensure-dotnet.sh"
# shellcheck disable=SC1091
source "$(dirname "$0")/activate-dotnet-offline.sh"

log "EL2 isolation: compile-only mode enabled (no runtime entrypoints invoked)"
log "building src/NfcReader/NfcReader.sln with --no-restore"

tme_run_segment "build-offline.compile" "${SEGMENT_MS}" -- dotnet build src/NfcReader/NfcReader.sln \
  --no-restore \
  -p:RestorePackages=false \
  -p:EnableEL2Runtime=false

log "build completed"
