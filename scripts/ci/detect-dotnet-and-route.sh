#!/usr/bin/env bash
set -euo pipefail

log() { printf '[router] %s\n' "$*"; }
err() { printf '[router][error] %s\n' "$*" >&2; }

if ! command -v dotnet >/dev/null 2>&1; then
  err 'dotnet missing on primary worker; expected in hardened runtime. route build to toolchain worker.'
  exit 127
fi

if ! dotnet --version >/tmp/dotnet-version.out 2>/tmp/dotnet-version.err; then
  rc=$?
  err "dotnet command exists but is unusable (exit ${rc}); route build to toolchain worker."
  if [[ -s /tmp/dotnet-version.err ]]; then
    err "dotnet diagnostics: $(tr '\n' ' ' </tmp/dotnet-version.err)"
  fi
  exit 127
fi

log "dotnet available on primary worker: $(cat /tmp/dotnet-version.out)"
log 'routing not required; primary worker can execute dotnet build directly.'
