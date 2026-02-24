#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export IATO_OPERATOR="${IATO_OPERATOR:-${USER:-unknown}}"
export IATO_PRNG_SEED="${IATO_PRNG_SEED:-0x4941544f2d5637}"

exec "${ROOT_DIR}/scripts/qemu-harness.sh" "$@"
