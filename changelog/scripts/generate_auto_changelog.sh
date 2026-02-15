#!/usr/bin/env bash
set -euo pipefail

CHANGELOG_FILE="${1:-changelog/output/AUTO_CHANGELOG.md}"
LATEST_TAG="$(git describe --tags --abbrev=0 2>/dev/null || true)"

mkdir -p "$(dirname "$CHANGELOG_FILE")"

{
  echo "# Changelog"
  echo
  if [[ -z "$LATEST_TAG" ]]; then
    echo "No tags found. Using full commit history."
    git log --pretty=format:"- %s (%h)"
  else
    echo "## [Unreleased] - $(date +%Y-%m-%d)"
    echo
    git log "$LATEST_TAG"..HEAD --pretty=format:"- %s (%h)"
    echo
    echo "## [$LATEST_TAG] - Previous Release"
    echo
    git log "$LATEST_TAG"^.."$LATEST_TAG" --pretty=format:"- %s (%h)"
  fi
} > "$CHANGELOG_FILE"

echo "Changelog generated at $CHANGELOG_FILE"
