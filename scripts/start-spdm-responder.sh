#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

IATO_SPDM_RESPONDER="${IATO_SPDM_RESPONDER:-${ROOT_DIR}/build/libspdm/bin/spdm_responder_emu}"
SPDM_SOCK="${IATO_SPDM_SOCK:-/tmp/iato-spdm.sock}"
SPDM_PID="${IATO_SPDM_PID:-/tmp/iato-spdm.pid}"
SPDM_LOG="${IATO_SPDM_LOG:-/tmp/iato-spdm.log}"
IATO_WAIT_LOOPS="${IATO_WAIT_LOOPS:-50}"

log() { echo "[spdm] $*"; }
err() { echo "[spdm][error] $*" >&2; }

command -v nc >/dev/null 2>&1 || { err "missing command: nc"; exit 1; }
[[ -x "${IATO_SPDM_RESPONDER}" ]] || { err "responder missing or not executable: ${IATO_SPDM_RESPONDER}"; exit 1; }

if [[ -f "${SPDM_PID}" ]]; then
  existing_pid="$(cat "${SPDM_PID}" 2>/dev/null || true)"
  if [[ -n "${existing_pid}" ]] && kill -0 "${existing_pid}" >/dev/null 2>&1; then
    log "already running (pid=${existing_pid})"
    exit 0
  fi
fi

rm -f "${SPDM_SOCK}" "${SPDM_PID}"
"${IATO_SPDM_RESPONDER}" --trans socket --socket-path "${SPDM_SOCK}" >>"${SPDM_LOG}" 2>&1 &
echo $! > "${SPDM_PID}"

for _ in $(seq 1 "${IATO_WAIT_LOOPS}"); do
  if nc -z -U "${SPDM_SOCK}" >/dev/null 2>&1; then
    log "ready: socket=${SPDM_SOCK} pid=$(cat "${SPDM_PID}")"
    exit 0
  fi
  sleep 0.1
done

err "responder did not become ready: ${SPDM_SOCK}"
exit 1
