#!/usr/bin/env bash
set -euo pipefail

SWTPM_DIR="${IATO_SWTPM_DIR:-/tmp/iato-swtpm}"
SWTPM_SOCK="${IATO_SWTPM_SOCK:-${SWTPM_DIR}/swtpm.sock}"
SWTPM_PID="${IATO_SWTPM_PID:-${SWTPM_DIR}/swtpm.pid}"
IATO_WAIT_LOOPS="${IATO_WAIT_LOOPS:-50}"

log() { echo "[swtpm] $*"; }
err() { echo "[swtpm][error] $*" >&2; }

command -v swtpm >/dev/null 2>&1 || { err "missing command: swtpm"; exit 1; }
command -v nc >/dev/null 2>&1 || { err "missing command: nc"; exit 1; }

if [[ -f "${SWTPM_PID}" ]]; then
  existing_pid="$(cat "${SWTPM_PID}" 2>/dev/null || true)"
  if [[ -n "${existing_pid}" ]] && kill -0 "${existing_pid}" >/dev/null 2>&1; then
    log "already running (pid=${existing_pid})"
    exit 0
  fi
fi

mkdir -p "${SWTPM_DIR}"
rm -f "${SWTPM_SOCK}" "${SWTPM_PID}"

swtpm socket \
  --tpmstate dir="${SWTPM_DIR}" \
  --ctrl type=unixio,path="${SWTPM_SOCK}" \
  --tpm2 --daemon --pid "${SWTPM_PID}"

for _ in $(seq 1 "${IATO_WAIT_LOOPS}"); do
  if nc -z -U "${SWTPM_SOCK}" >/dev/null 2>&1; then
    log "ready: socket=${SWTPM_SOCK} pid=$(cat "${SWTPM_PID}")"
    exit 0
  fi
  sleep 0.1
done

err "swtpm did not become ready: ${SWTPM_SOCK}"
exit 1
