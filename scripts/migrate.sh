#!/usr/bin/env bash
set -euo pipefail

mkdir -p workers/modern
cp -n data/reference.json workers/modern/reference.json || true
echo "Migration scaffold complete"
