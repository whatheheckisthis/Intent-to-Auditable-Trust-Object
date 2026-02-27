#!/usr/bin/env bash
set -euo pipefail

required_tools=(python3 git openssl)
optional_tools=(conda flake8 docker podman)
shell_candidates=(bash sh)
missing_required=()

log() { echo "[verify_environment] $*"; }

for tool in "${required_tools[@]}"; do
  if command -v "$tool" >/dev/null 2>&1; then
    log "found required tool: $tool"
  else
    log "missing required tool: $tool"
    missing_required+=("$tool")
  fi
done

for tool in "${optional_tools[@]}"; do
  if command -v "$tool" >/dev/null 2>&1; then
    log "found optional tool: $tool"
  else
    log "optional tool not found: $tool"
  fi
done

runtime="none"
if command -v docker >/dev/null 2>&1; then
  runtime="docker"
elif command -v podman >/dev/null 2>&1; then
  runtime="podman"
fi
log "selected container runtime: $runtime"

selected_shell="none"
for candidate in "${shell_candidates[@]}"; do
  if command -v "$candidate" >/dev/null 2>&1; then
    selected_shell="$candidate"
    break
  fi
done
log "selected shell: $selected_shell"

if [[ ${#missing_required[@]} -gt 0 ]]; then
  log "error: required tools missing -> ${missing_required[*]}"
  exit 1
fi

python3 --version
log "environment verification complete"
