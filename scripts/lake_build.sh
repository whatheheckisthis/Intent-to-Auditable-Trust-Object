#!/usr/bin/env bash
set -euo pipefail

cd lean
echo "Building IATO-V7 formal verification..."
lake update
lake build
lake build tests
echo "Build complete"
