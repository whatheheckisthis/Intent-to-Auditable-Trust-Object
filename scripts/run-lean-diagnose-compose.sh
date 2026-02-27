#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="$ROOT_DIR/docker/lean-diagnose/docker-compose.yml"

if command -v docker >/dev/null 2>&1; then
  echo "[compose] using docker compose"
  docker compose -f "$COMPOSE_FILE" up -d --build
  echo "[compose] lean diagnose UI via reverse proxy: http://127.0.0.1:8088/"
  echo "[compose] health check: http://127.0.0.1:8088/healthz"
  exit 0
fi

if command -v podman >/dev/null 2>&1; then
  echo "[compose] using podman compose"
  podman compose -f "$COMPOSE_FILE" up -d --build
  echo "[compose] lean diagnose UI via reverse proxy: http://127.0.0.1:8088/"
  echo "[compose] health check: http://127.0.0.1:8088/healthz"
  exit 0
fi

echo "[compose] neither docker nor podman is installed"
exit 127
