#!/usr/bin/env bash
set -euo pipefail

echo "Running environment verification..."

required_tools=(openssl git pandoc python3)
for tool in "${required_tools[@]}"; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "ERROR: required tool '$tool' is not installed or not in PATH"
    exit 1
  fi
  echo "OK: $tool"
done

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda is not available"
  exit 1
fi

echo "OK: conda"
echo "Environment verification passed."
