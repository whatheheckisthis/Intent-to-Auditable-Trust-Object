#!/usr/bin/env bash
set -euo pipefail

CHANGELOG_FILE="${1:-CHANGELOG.md}"
OUTPUT_FILE="${2:-changelog/output/changelog_summary.txt}"

if [[ ! -f "$CHANGELOG_FILE" ]]; then
  echo "ERROR: changelog not found at $CHANGELOG_FILE"
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT_FILE")"

awk '
BEGIN { RS = "" }
/^## \[.*\]/ {
  split($1, v, "[\\[\\]]")
  print "Version: " v[2]
  print "----------------------"
  print $0
  print ""
}
' "$CHANGELOG_FILE" > "$OUTPUT_FILE"

echo "Summary written to $OUTPUT_FILE"
