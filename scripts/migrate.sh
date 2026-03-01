#!/usr/bin/env bash
set -euo pipefail

mkdir -p workers/target
cp -n data/reference.json workers/target/reference.json || true
echo "Migration scaffold complete"
