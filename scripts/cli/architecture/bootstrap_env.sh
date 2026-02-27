#!/usr/bin/env bash
set -euo pipefail

STRICT_MODE="${BOOTSTRAP_STRICT:-0}"
INSTALL_WORKER_PACKAGES="${BOOTSTRAP_INSTALL_WORKER_PACKAGES:-0}"
INSTALL_CONTAINER_PACKAGES="${BOOTSTRAP_INSTALL_CONTAINER_PACKAGES:-0}"
PREFERRED_RUNTIME="${BOOTSTRAP_CONTAINER_RUNTIME:-auto}"
PREFERRED_SHELL="${BOOTSTRAP_SHELL:-auto}"
RUNTIME_ENV_FILE="${BOOTSTRAP_RUNTIME_ENV_FILE:-ci/tools/.env.runtime}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --strict)
      STRICT_MODE=1
      ;;
    --install-worker-packages)
      INSTALL_WORKER_PACKAGES=1
      ;;
    --install-container-packages)
      INSTALL_CONTAINER_PACKAGES=1
      ;;
    --container-runtime)
      shift
      PREFERRED_RUNTIME="${1:-auto}"
      ;;
    --shell)
      shift
      PREFERRED_SHELL="${1:-auto}"
      ;;
    *)
      echo "[bootstrap_env][error] unknown argument: $1" >&2
      exit 2
      ;;
  esac
  shift
done

log() { echo "[bootstrap_env] $*"; }
warn() { echo "[bootstrap_env][warn] $*" >&2; }

run_with_log() {
  local label="$1"
  local cmd="$2"
  local log_file
  log_file="$(mktemp)"

  if eval "$cmd" >"$log_file" 2>&1; then
    rm -f "$log_file"
    return 0
  fi

  warn "${label} failed"
  warn "recent output:"
  tail -n 20 "$log_file" | sed 's/^/[bootstrap_env][detail] /'
  rm -f "$log_file"

  if [[ "$STRICT_MODE" == "1" ]]; then
    warn "strict mode enabled; failing bootstrap"
    return 1
  fi

  warn "continuing without failing (set BOOTSTRAP_STRICT=1 to enforce)"
  return 0
}

ensure_packages() {
  local kind="$1"
  shift
  local required=("$@")
  local missing=()

  for pkg in "${required[@]}"; do
    command -v "$pkg" >/dev/null 2>&1 || missing+=("$pkg")
  done

  if [[ ${#missing[@]} -eq 0 ]]; then
    log "${kind} prerequisites already satisfied"
    return 0
  fi

  warn "missing ${kind} packages: ${missing[*]}"

  if command -v apt-get >/dev/null 2>&1; then
    local installer="apt-get"
    local install_targets=()
    if [[ "${EUID:-$(id -u)}" -ne 0 ]] && command -v sudo >/dev/null 2>&1; then
      installer="sudo apt-get"
    fi

    for item in "${missing[@]}"; do
      if [[ "$item" == "docker" ]]; then
        install_targets+=("docker.io")
      else
        install_targets+=("$item")
      fi
    done

    run_with_log "apt update" "$installer update"
    run_with_log "apt install" "$installer install -y ${install_targets[*]}"
    return 0
  fi

  if command -v yum >/dev/null 2>&1; then
    local installer="yum"
    local install_targets=()
    if [[ "${EUID:-$(id -u)}" -ne 0 ]] && command -v sudo >/dev/null 2>&1; then
      installer="sudo yum"
    fi

    for item in "${missing[@]}"; do
      if [[ "$item" == "docker" ]]; then
        install_targets+=("docker-ce")
      else
        install_targets+=("$item")
      fi
    done

    run_with_log "yum install" "$installer install -y ${install_targets[*]}"
    return 0
  fi

  warn "no supported package manager detected; install missing packages manually"
  [[ "$STRICT_MODE" == "1" ]] && return 1 || return 0
}

resolve_shell() {
  if [[ "$PREFERRED_SHELL" != "auto" ]]; then
    if command -v "$PREFERRED_SHELL" >/dev/null 2>&1; then
      echo "$PREFERRED_SHELL"
      return 0
    fi
    warn "requested shell '$PREFERRED_SHELL' is unavailable"
    [[ "$STRICT_MODE" == "1" ]] && return 1
  fi

  if command -v bash >/dev/null 2>&1; then
    echo "bash"
    return 0
  fi
  if command -v sh >/dev/null 2>&1; then
    echo "sh"
    return 0
  fi

  warn "no compatible shell found"
  [[ "$STRICT_MODE" == "1" ]] && return 1 || { echo "none"; return 0; }
}

resolve_runtime() {
  if [[ "$PREFERRED_RUNTIME" != "auto" ]]; then
    if command -v "$PREFERRED_RUNTIME" >/dev/null 2>&1; then
      echo "$PREFERRED_RUNTIME"
      return 0
    fi
    warn "requested runtime '$PREFERRED_RUNTIME' is unavailable"
    [[ "$STRICT_MODE" == "1" ]] && return 1
  fi

  if command -v docker >/dev/null 2>&1; then
    echo "docker"
    return 0
  fi
  if command -v podman >/dev/null 2>&1; then
    echo "podman"
    return 0
  fi

  warn "no container runtime found (docker/podman)"
  [[ "$STRICT_MODE" == "1" ]] && return 1 || { echo "none"; return 0; }
}

if [[ "$INSTALL_WORKER_PACKAGES" == "1" ]]; then
  log "installing worker system packages"
  ensure_packages "worker" git curl unzip jq
fi

if [[ "$INSTALL_CONTAINER_PACKAGES" == "1" ]]; then
  log "installing container runtime packages"
  ensure_packages "container" docker podman
fi

SELECTED_SHELL="$(resolve_shell)"
SELECTED_RUNTIME="$(resolve_runtime)"

mkdir -p "$(dirname "$RUNTIME_ENV_FILE")"
cat > "$RUNTIME_ENV_FILE" <<RUNTIME_ENV
BOOTSTRAP_SELECTED_SHELL=${SELECTED_SHELL}
BOOTSTRAP_SELECTED_CONTAINER_RUNTIME=${SELECTED_RUNTIME}
RUNTIME_ENV
log "wrote runtime integration env file: $RUNTIME_ENV_FILE"

if command -v conda >/dev/null 2>&1; then
  log "conda detected; ensuring testenv exists"
  run_with_log "conda create testenv" "conda create -y -n testenv python=3.11"

  log "installing baseline Python tooling into conda env"
  run_with_log "conda pip upgrade" "conda run -n testenv python -m pip install --disable-pip-version-check --upgrade pip"
  run_with_log "conda package bootstrap" "conda run -n testenv python -m pip install --disable-pip-version-check -e .[test] flake8"
  log "conda environment bootstrap complete"
else
  log "conda not found; bootstrapping with current interpreter"
  run_with_log "pip upgrade" "python3 -m pip install --disable-pip-version-check --upgrade pip"
  run_with_log "package bootstrap" "python3 -m pip install --disable-pip-version-check -e .[test] flake8"
  log "fallback bootstrap complete"
fi
