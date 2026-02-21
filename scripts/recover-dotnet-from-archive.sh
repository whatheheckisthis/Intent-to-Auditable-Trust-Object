#!/usr/bin/env bash
# Deterministic offline .NET SDK recovery for restricted CI workers.
set -euo pipefail

EXIT_NO_ARCHIVE=42
EXIT_UNSUPPORTED_ARCH=43
EXIT_RECOVERY_FAILED=44
SEGMENT_MS="${TME_SEGMENT_RECOVER_DOTNET_MS:-5200}"

# shellcheck disable=SC1091
source "$(dirname "$0")/timing_mitigation_engine.sh"

log() { printf '[dotnet-recover] %s\n' "$*"; }
fail() {
  local code="$1"; shift
  printf '[dotnet-recover][error] %s\n' "$*" >&2
  exit "${code}"
}

read_global_json_value() {
  local key="$1"
  local global_json="src/NfcReader/global.json"

  [[ -f "${global_json}" ]] || fail "${EXIT_NO_ARCHIVE}" "required file not found: ${global_json}"

  awk -F'"' -v key="${key}" '$2 == key { print $4; exit }' "${global_json}"
}

# Resolve architecture naming used by staged archive file names.
detect_arch() {
  local m
  m="$(uname -m)"
  case "${m}" in
    x86_64|amd64) echo "linux-x64" ;;
    aarch64|arm64) echo "linux-arm64" ;;
    *) fail "${EXIT_UNSUPPORTED_ARCH}" "unsupported architecture: ${m} (expected x86_64/aarch64)" ;;
  esac
}

is_root() {
  [[ "$(id -u)" -eq 0 ]]
}

# Candidate archive locations (no network access, local-only).
find_archive() {
  local arch="$1"
  local explicit="${DOTNET_ARCHIVE:-}"
  if [[ -n "${explicit}" ]]; then
    [[ -f "${explicit}" ]] || fail "${EXIT_NO_ARCHIVE}" "DOTNET_ARCHIVE is set but file not found: ${explicit}"
    echo "${explicit}"
    return
  fi

  local roots=(
    "/opt/bootstrap"
    "${HOME}/bootstrap"
    "/workspace/bootstrap"
    "${DOTNET_INSTALL_DIR:-}"
    "${HOME}/.dotnet"
    "/usr/local/dotnet"
    "/usr/share/dotnet"
  )
  local candidate
  for root in "${roots[@]}"; do
    [[ -n "${root}" && -d "${root}" ]] || continue
    while IFS= read -r -d '' candidate; do
      echo "${candidate}"
      return
    done < <(find "${root}" -maxdepth 1 -type f -name "dotnet-sdk-*-${arch}.tar.gz" -print0 | sort -z)
  done

  return 1
}

verify_checksum_if_present() {
  local archive="$1"
  local checksum_file="${archive}.sha512"
  if [[ ! -f "${checksum_file}" ]]; then
    log "no checksum found (${checksum_file}); skipping integrity verification"
    return
  fi

  command -v sha512sum >/dev/null 2>&1 || fail "${EXIT_RECOVERY_FAILED}" "checksum file present but sha512sum is unavailable"

  log "verifying checksum via ${checksum_file}"
  (cd "$(dirname "${archive}")" && sha512sum -c "$(basename "${checksum_file}")") || fail "${EXIT_RECOVERY_FAILED}" "checksum verification failed for ${archive}"
}

configure_persistent_env() {
  local dotnet_root="$1"
  local profile_file
  if is_root; then
    profile_file="/etc/profile.d/dotnet.sh"
    cat > "${profile_file}" <<PROFILE_EOF
export DOTNET_ROOT="${dotnet_root}"
export PATH="${dotnet_root}:${dotnet_root}/tools:\$PATH"
PROFILE_EOF
    chmod 0644 "${profile_file}"
    log "wrote system profile: ${profile_file}"
  else
    profile_file="${HOME}/.bashrc"
    if ! grep -q 'DOTNET_ROOT="'"${dotnet_root}"'"' "${profile_file}" 2>/dev/null; then
      cat >> "${profile_file}" <<PROFILE_EOF

# Added by scripts/recover-dotnet-from-archive.sh
export DOTNET_ROOT="${dotnet_root}"
export PATH="${dotnet_root}:${dotnet_root}/tools:\$PATH"
PROFILE_EOF
    fi
    log "updated user profile: ${profile_file}"
  fi
}

ensure_runtime_visible() {
  local dotnet_root="$1"
  export DOTNET_ROOT="${dotnet_root}"
  export PATH="${dotnet_root}:${dotnet_root}/tools:${PATH}"
}

main() {
  local arch archive target_root version rid
  arch="$(detect_arch)"
  log "detected architecture: ${arch}"

  if ! archive="$(find_archive "${arch}")"; then
    version="$(read_global_json_value "version")"
    rid="$(read_global_json_value "rid")"
    printf '[dotnet-recover][error] no staged SDK archive found for arch=%s\n' "${arch}" >&2
    printf '[dotnet-recover][error] to fix: download dotnet-sdk-%s-%s.tar.gz from https://dotnet.microsoft.com/download/dotnet and place it in any of the following paths:\n' "${version}" "${rid}" >&2
    printf '[dotnet-recover][error]   %s\n' "/opt/bootstrap" >&2
    printf '[dotnet-recover][error]   %s\n' "${HOME}/bootstrap" >&2
    printf '[dotnet-recover][error]   %s\n' "/workspace/bootstrap" >&2
    if [[ -n "${DOTNET_INSTALL_DIR:-}" ]]; then
      printf '[dotnet-recover][error]   %s\n' "${DOTNET_INSTALL_DIR}" >&2
    fi
    printf '[dotnet-recover][error]   %s\n' "${HOME}/.dotnet" >&2
    printf '[dotnet-recover][error]   %s\n' "/usr/local/dotnet" >&2
    printf '[dotnet-recover][error]   %s\n' "/usr/share/dotnet" >&2
    exit "${EXIT_NO_ARCHIVE}"
  fi
  log "using archive: ${archive}"

  if is_root; then
    target_root="/opt/dotnet"
  else
    target_root="${HOME}/.dotnet"
  fi
  mkdir -p "${target_root}"

  verify_checksum_if_present "${archive}"

  log "extracting SDK into ${target_root}"
  tar -zxf "${archive}" -C "${target_root}" || fail "${EXIT_RECOVERY_FAILED}" "failed to extract archive: ${archive}"

  configure_persistent_env "${target_root}"
  ensure_runtime_visible "${target_root}"

  command -v dotnet >/dev/null 2>&1 || fail "${EXIT_RECOVERY_FAILED}" "dotnet binary not discoverable after extraction"
  dotnet --info >/dev/null 2>&1 || fail "${EXIT_RECOVERY_FAILED}" "dotnet binary exists but is not executable"

  log "recovery complete; dotnet version: $(dotnet --version)"
}

tme_run_segment "recover-dotnet-from-archive" "${SEGMENT_MS}" -- main "$@"
