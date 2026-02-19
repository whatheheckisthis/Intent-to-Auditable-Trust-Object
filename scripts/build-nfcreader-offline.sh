#!/usr/bin/env bash
# Build-staging workflow: compile-only validation, no restore/runtime execution.
set -euo pipefail

log() { printf '[build-offline] %s\n' "$*"; }
fail() { printf '[build-offline][error] %s\n' "$*" >&2; exit 46; }

bash "$(dirname "$0")/ensure-dotnet.sh"
# shellcheck disable=SC1091
source "$(dirname "$0")/activate-dotnet-offline.sh"

log "EL2 isolation: compile-only mode enabled (no runtime entrypoints invoked)"
log "building src/NfcReader/NfcReader.sln with --no-restore"

dotnet build src/NfcReader/NfcReader.sln \
  --no-restore \
  -p:RestorePackages=false \
  -p:EnableEL2Runtime=false

log "build completed"
