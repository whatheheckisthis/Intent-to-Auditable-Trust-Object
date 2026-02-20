#!/usr/bin/env bash
# Deterministic timing mitigation engine for offline bootstrap/build workflows.
# Source this file and wrap critical segments via tme_run_segment.
#
# Key behavior:
# - Pins a segment to a constant execution window in milliseconds.
# - Uses CPU-bound deterministic padding (no sleep) when a segment completes early.
# - Uses a deterministic PRNG to add reproducible micro-variation to padding workload.
# - Supports simulation mode (measure + log only) via TME_SIMULATION_MODE=1.
#
# Environment controls:
#   TME_SEED                  Seed material for deterministic PRNG (default: intent-ato-seed)
#   TME_SIMULATION_MODE       1 = observe only, no padding/overrun adjustment
#   TME_STRICT_OVERRUN        1 = force deterministic overrun exit code on window breach
#   TME_OVERRUN_EXIT_CODE     Exit code used in strict overrun mode (default: 47)
#   TME_LOG_FILE              Optional file to append mitigation logs

if [[ -n "${_TME_ENGINE_LOADED:-}" ]]; then
  return 0 2>/dev/null || exit 0
fi
readonly _TME_ENGINE_LOADED=1

: "${TME_SEED:=intent-ato-seed}"
: "${TME_SIMULATION_MODE:=0}"
: "${TME_STRICT_OVERRUN:=0}"
: "${TME_OVERRUN_EXIT_CODE:=47}"
: "${TME_INVOCATION_ID:=0}"

_tme_log() {
  local level="$1"; shift
  local msg="[timing-mitigation][${level}] $*"
  printf '%s\n' "${msg}" >&2
  if [[ -n "${TME_LOG_FILE:-}" ]]; then
    printf '%s\n' "${msg}" >> "${TME_LOG_FILE}"
  fi
}

tme_now_ms() {
  # EPOCHREALTIME is bash-native: seconds.microseconds
  local sec usec
  sec="${EPOCHREALTIME%.*}"
  usec="${EPOCHREALTIME#*.}"
  usec="${usec:0:6}"
  printf '%d\n' "$((10#${sec} * 1000 + 10#${usec} / 1000))"
}

_tme_seed_to_u64() {
  local src="$1"
  local c i val=1469598103934665603
  for ((i=0; i<${#src}; i++)); do
    c=$(printf '%d' "'${src:i:1}")
    val=$(( (val ^ c) * 1099511628211 ))
  done
  printf '%d\n' "$(( val & 0x7FFFFFFFFFFFFFFF ))"
}

tme_prng_init() {
  local segment="$1"
  TME_INVOCATION_ID=$((TME_INVOCATION_ID + 1))
  TME_PRNG_STATE="$(_tme_seed_to_u64 "${TME_SEED}|${segment}|${TME_INVOCATION_ID}")"
}

tme_prng_next() {
  # LCG (deterministic and lightweight for scheduling micro-variation).
  TME_PRNG_STATE=$(( (TME_PRNG_STATE * 6364136223846793005 + 1442695040888963407) & 0x7FFFFFFFFFFFFFFF ))
  printf '%d\n' "${TME_PRNG_STATE}"
}

tme_cpu_pad_until_ms() {
  local deadline_ms="$1"
  local entropy rounds i mix=0 rand now
  while :; do
    now="$(tme_now_ms)"
    (( now >= deadline_ms )) && break

    rand="$(tme_prng_next)"
    rounds=$(( (rand & 1023) + 256 ))
    for ((i=0; i<rounds; i++)); do
      mix=$(( (mix * 1664525 + 1013904223 + i + (rand & 255)) & 0xFFFFFFFF ))
    done
  done
  : "$mix"
}

# Usage:
#   tme_run_segment <segment_name> <constant_ms> -- <command> [args...]
tme_run_segment() {
  local segment="$1"
  local constant_ms="$2"
  shift 2
  [[ "${1:-}" == "--" ]] && shift

  local start_ms elapsed_ms end_ms rc=0 overrun=0
  tme_prng_init "${segment}"
  start_ms="$(tme_now_ms)"

  "$@" || rc=$?

  end_ms="$(tme_now_ms)"
  elapsed_ms=$(( end_ms - start_ms ))

  if [[ "${TME_SIMULATION_MODE}" == "1" ]]; then
    _tme_log "sim" "segment=${segment} rc=${rc} actual_ms=${elapsed_ms} target_ms=${constant_ms}"
    return "${rc}"
  fi

  if (( elapsed_ms < constant_ms )); then
    tme_cpu_pad_until_ms "$((start_ms + constant_ms))"
    end_ms="$(tme_now_ms)"
    elapsed_ms=$(( end_ms - start_ms ))
  elif (( elapsed_ms > constant_ms )); then
    overrun=1
    if [[ "${TME_STRICT_OVERRUN}" == "1" ]]; then
      rc="${TME_OVERRUN_EXIT_CODE}"
    fi
  fi

  _tme_log "audit" "segment=${segment} rc=${rc} actual_ms=${elapsed_ms} target_ms=${constant_ms} overrun=${overrun}"
  return "${rc}"
}

# Deterministic exit code helper for scripts that need explicit audited exits.
tme_exit_code() {
  local code="$1"; shift
  _tme_log "exit" "code=${code} reason=$*"
  return "${code}"
}
