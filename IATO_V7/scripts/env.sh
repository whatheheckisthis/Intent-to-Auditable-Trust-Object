# shellcheck shell=bash
# Source this file before running lake/podman commands:
#   source scripts/env.sh

export ELAN_HOME="${ELAN_HOME:-$HOME/.elan}"
export PATH="$ELAN_HOME/bin:$PATH"
export LAKE="$ELAN_HOME/bin/lake"

# Lean/Lake build defaults
export LEAN_ABORT_ON_PANIC="${LEAN_ABORT_ON_PANIC:-1}"
export LEAN_PATH="${LEAN_PATH:-$PWD/.lake/packages}"
export LAKE_NO_CACHE="${LAKE_NO_CACHE:-0}"

# Podman defaults for reproducible local builds and fewer UID mapping issues
export PODMAN_USERNS="${PODMAN_USERNS:-keep-id}"
export PODMAN_SYSTEMD_UNIT="${PODMAN_SYSTEMD_UNIT:-false}"
export COMPOSE_DOCKER_CLI_BUILD="${COMPOSE_DOCKER_CLI_BUILD:-1}"
