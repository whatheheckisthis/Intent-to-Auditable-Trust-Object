#!/usr/bin/env bash
#
# move_to_legacy.sh <filename>
#
# Moves a single file to bin/legacy/ with Git tracking.

set -e

# Check if filename is provided
if [ -z "$1" ]; then
  echo "Usage: bash move_to_legacy.sh <filename>"
  exit 1
fi

FILE="$1"
DEST="bin/legacy"

if [[ "$VIRTUAL_ENV" != "" ]]; then
  echo "Virtual environment is active: $VIRTUAL_ENV"
else
  echo "No virtual environment detected"
fi

