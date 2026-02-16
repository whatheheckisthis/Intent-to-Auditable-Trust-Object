#!/bin/bash
# Convert markdown to HTML

INPUT=$1
OUTPUT="${INPUT%.md}.html"

if [ -z "$INPUT" ]; then
  echo "Usage: ./md_to_html.sh <file.md>"
  exit 1
fi

pandoc "$INPUT" -o "$OUTPUT" && echo "Converted to $OUTPUT"
