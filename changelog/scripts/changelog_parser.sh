#!/usr/bin/env bash
set -euo pipefail

CHANGELOG_FILE="${1:-CHANGELOG.md}"
OUTPUT_FILE="${2:-changelog/output/parsed_changelog.txt}"

if [[ ! -f "$CHANGELOG_FILE" ]]; then
  echo "ERROR: changelog not found at $CHANGELOG_FILE"
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT_FILE")"
grep -E '^##\s*v?[0-9]+\.[0-9]+\.[0-9]+' "$CHANGELOG_FILE" > "$OUTPUT_FILE"
cat "$OUTPUT_FILE"
