#!/usr/bin/env bash
# Activate an SDK-only .NET toolchain without network access.
set -euo pipefail

EXIT_ACTIVATION_FAILED=42

ok() { printf '[activate-dotnet] %s\n' "$*" >&2; }
error() { printf '[activate-dotnet][error] %s\n' "$*" >&2; }

activation_exit() {
  local code="$1"
  if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    return "${code}"
  fi
  exit "${code}"
}

read_required_version() {
  local global_json="src/NfcReader/global.json"
  awk -F'"' '$2 == "version" { print $4; exit }' "${global_json}"
}

detect_dotnet_root() {
  local roots=(
    "/opt/bootstrap"
    "${HOME}/bootstrap"
    "/workspace/bootstrap"
    "${DOTNET_INSTALL_DIR:-}"
    "${HOME}/.dotnet"
    "/usr/local/dotnet"
    "/usr/share/dotnet"
  )
  local root

  if [[ -n "${DOTNET_ROOT:-}" ]]; then
    echo "${DOTNET_ROOT}"
    return 0
  fi

  for root in "${roots[@]}"; do
    [[ -n "${root}" ]] || continue
    if [[ -x "${root}/dotnet" ]]; then
      echo "${root}"
      return 0
    fi
  done

  return 1
}

main() {
  local required_version found_version dotnet_root dotnet_path
  required_version="$(read_required_version)"

  if dotnet_root="$(detect_dotnet_root)"; then
    export DOTNET_ROOT="${dotnet_root}"
    dotnet_path="${DOTNET_ROOT}/dotnet"
  else
    found_version="not-found"
    dotnet_path="not-found"
    error "activation failed: dotnet ${found_version} at ${dotnet_path}, required ${required_version}"
    activation_exit "${EXIT_ACTIVATION_FAILED}"
  fi

  export PATH="${DOTNET_ROOT}:${DOTNET_ROOT}/tools:${PATH}"
  export DOTNET_CLI_TELEMETRY_OPTOUT=1
  export DOTNET_SKIP_FIRST_TIME_EXPERIENCE=1
  export DOTNET_NOLOGO=1

  if ! found_version="$(dotnet --version 2>/dev/null)"; then
    found_version="not-found"
  fi

  if [[ "${found_version}" != "${required_version}" ]]; then
    error "activation failed: dotnet ${found_version} at ${dotnet_path}, required ${required_version}"
    activation_exit "${EXIT_ACTIVATION_FAILED}"
  fi

  ok "OK: dotnet ${required_version} active at ${DOTNET_ROOT}"
  activation_exit 0
}

main "$@"
