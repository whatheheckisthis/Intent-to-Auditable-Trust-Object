#!/bin/bash
# Add timestamps to lines from tail -f

FILE=$1

if [ -z "$FILE" ]; then
  echo "Usage: ./timestamp_tail.sh <file_to_tail>"
  exit 1
fi

tail -f "$FILE" | while read line; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $line"
done
