#!/bin/bash
#
# scripts/tag_release.sh
#
# Purpose:
#   Tag the current Git commit with a semantic version and optional message.
#
# Usage:
#   ./scripts/tag_release.sh v0.2.0 "Release v0.2.0 – Production ready"
#
# Notes:
#   - This script creates an annotated Git tag.
#   - Pushes the tag to origin if the --push flag is provided.
#   - Requires Git to be initialized and clean.

VERSION=$1
MESSAGE=$2
PUSH_FLAG=$3

if [ -z "$VERSION" ]; then
  echo "Usage: $0 <version> [message] [--push]"
  exit 1
fi

# Set default message if not provided
if [ -z "$MESSAGE" ]; then
  MESSAGE="Release $VERSION"
fi

echo "Tagging commit with version: $VERSION"
git tag -a "$VERSION" -m "$MESSAGE"

if [ "$PUSH_FLAG" == "--push" ]; then
  echo "Pushing tag to origin..."
  git push origin "$VERSION"
fi

echo "Tag created: $VERSION"

# End of script
