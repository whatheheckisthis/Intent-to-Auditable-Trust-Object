#!/usr/bin/env bash
set -e

echo "[*] Bootstrapping ops-core container..."

mkdir -p /app/config/keys
mkdir -p /app/data

echo "[*] Container is ready. Drop into shell or run scripts."
exec "$@"