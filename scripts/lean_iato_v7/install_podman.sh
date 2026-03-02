#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/env.sh"

if command -v podman >/dev/null 2>&1 && command -v podman-compose >/dev/null 2>&1; then
  echo "podman + podman-compose already available"
  podman --version || true
  podman-compose --version || true
  exit 0
fi

echo "Installing podman + podman-compose..."
if ! command -v apt-get >/dev/null 2>&1; then
  echo "apt-get not available on this system"
  exit 1
fi

set +e
apt-get update
apt_status=$?
set -e
if [ "$apt_status" -ne 0 ]; then
  echo "apt-get update failed (often proxy/network restriction)."
  echo "Current proxy env (if set):"
  env | rg -i 'proxy' || true
  exit "$apt_status"
fi

apt-get install -y podman podman-compose uidmap slirp4netns fuse-overlayfs

if ! command -v podman >/dev/null 2>&1; then
  echo "podman installation appears incomplete"
  exit 1
fi

# Ensure env helper has podman defaults used by this project.
if ! grep -q 'PODMAN_USERNS=' "$ENV_FILE" 2>/dev/null; then
  cat >> "$ENV_FILE" <<'ENV'

# Podman defaults for reproducible local builds
export PODMAN_USERNS="keep-id"
export PODMAN_SYSTEMD_UNIT="false"
export COMPOSE_DOCKER_CLI_BUILD="1"
ENV
  echo "Updated $ENV_FILE with Podman-related environment defaults."
fi

echo "podman installed: $(command -v podman)"
podman --version || true
command -v podman-compose >/dev/null 2>&1 && podman-compose --version || true

echo "Done. Next steps:"
echo "  source scripts/env.sh"
echo "  podman-compose -f podman-compose.yml up --abort-on-container-exit"
