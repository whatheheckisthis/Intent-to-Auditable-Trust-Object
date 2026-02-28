#!/usr/bin/env bash
set -euo pipefail

mkdir -p workers/new
cp -n data/reference.json workers/new/reference.json || true
echo "Migration scaffold complete"
