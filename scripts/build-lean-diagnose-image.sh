#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="lean-diagnose:local"
DOCKERFILE="docker/lean-diagnose/Dockerfile"

if command -v docker >/dev/null 2>&1; then
  echo "[build] using docker"
  docker build -t "$IMAGE_NAME" -f "$DOCKERFILE" .
  echo "[build] built $IMAGE_NAME"
  exit 0
fi

if command -v podman >/dev/null 2>&1; then
  echo "[build] using podman"
  podman build -t "$IMAGE_NAME" -f "$DOCKERFILE" .
  echo "[build] built $IMAGE_NAME"
  exit 0
fi

echo "[build] neither docker nor podman is installed in this environment"
echo "[build] run ./scripts/setup-lean-diagnose-runtime.sh to install/configure"
exit 127
