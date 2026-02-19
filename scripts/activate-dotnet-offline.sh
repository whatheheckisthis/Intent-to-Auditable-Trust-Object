#!/usr/bin/env bash
# Activate an SDK-only .NET toolchain without network access.
set -euo pipefail

log() { printf '[activate-dotnet] %s\n' "$*"; }
fail() { printf '[activate-dotnet][error] %s\n' "$*" >&2; exit 45; }

if [[ -x /opt/dotnet/dotnet ]]; then
  export DOTNET_ROOT=/opt/dotnet
elif [[ -x "${HOME}/.dotnet/dotnet" ]]; then
  export DOTNET_ROOT="${HOME}/.dotnet"
else
  fail "no recovered SDK found at /opt/dotnet or ${HOME}/.dotnet; run scripts/ensure-dotnet.sh first"
fi

export PATH="${DOTNET_ROOT}:${DOTNET_ROOT}/tools:${PATH}"
export DOTNET_CLI_TELEMETRY_OPTOUT=1
export DOTNET_NOLOGO=1

# Force deterministic local behavior in restricted workers.
export NUGET_PACKAGES="${NUGET_PACKAGES:-${PWD}/.nuget/packages}"

command -v dotnet >/dev/null 2>&1 || fail "dotnet not discoverable after activation"
dotnet --info >/dev/null 2>&1 || fail "dotnet is not executable after activation"

log "activated SDK at ${DOTNET_ROOT}"
log "dotnet version: $(dotnet --version)"
