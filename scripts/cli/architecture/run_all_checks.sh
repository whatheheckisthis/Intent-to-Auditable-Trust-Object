#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log() { echo "[run_all_checks] $*"; }

run_step() {
  local step_name="$1"
  local step_cmd="$2"

  log "starting: ${step_name}"
  if eval "$step_cmd"; then
    log "completed: ${step_name}"
  else
    local exit_code=$?
    log "failed: ${step_name} (exit ${exit_code})"
    return "$exit_code"
  fi
}

bootstrap_args=()
if [[ "${BOOTSTRAP_INSTALL_WORKER_PACKAGES:-0}" == "1" ]]; then
  bootstrap_args+=("--install-worker-packages")
fi
if [[ "${BOOTSTRAP_INSTALL_CONTAINER_PACKAGES:-0}" == "1" ]]; then
  bootstrap_args+=("--install-container-packages")
fi
if [[ "${BOOTSTRAP_STRICT:-0}" == "1" ]]; then
  bootstrap_args+=("--strict")
fi
if [[ -n "${BOOTSTRAP_CONTAINER_RUNTIME:-}" ]]; then
  bootstrap_args+=("--container-runtime" "${BOOTSTRAP_CONTAINER_RUNTIME}")
fi
if [[ -n "${BOOTSTRAP_SHELL:-}" ]]; then
  bootstrap_args+=("--shell" "${BOOTSTRAP_SHELL}")
fi

run_step "verify environment" "bash '$SCRIPT_DIR/verify_environment.sh'"
run_step "bootstrap environment" "bash '$SCRIPT_DIR/bootstrap_env.sh' ${bootstrap_args[*]}"
run_step "verify imports" "python3 '$SCRIPT_DIR/verify_imports.py'"
run_step "lint" "bash '$SCRIPT_DIR/lint.sh'"

log "all checks passed"
