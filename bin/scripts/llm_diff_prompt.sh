#!/bin/bash

# Usage: ./llm_diff_prompt.sh path/to/file.txt

INPUT_FILE="$1"

if [ -z "$INPUT_FILE" ] || [ ! -f "$INPUT_FILE" ]; then
  echo "⚠️  Please provide a valid file path."
  exit 1
fi

echo "🔍 Analyzing file: $INPUT_FILE"

# Extract patterns
DATES=$(grep -Eo '[0-9]{2,4}[-/][0-9]{2}[-/][0-9]{2,4}' "$INPUT_FILE" | sort -u)
HARDCODED=$(grep -E 'TODO|FIXME|temp|hardcoded|v[0-9]+\.[0-9]+' "$INPUT_FILE" | sort -u | head -n 10)
PLACEHOLDERS=$(grep -E '\[.*\]|\{.*\}' "$INPUT_FILE" | sort -u | head -n 10)

echo
echo "----------------------------------------------------"
echo "🧠 Copy the following LLM prompt into ChatGPT:"
echo "----------------------------------------------------"
echo
echo '"""'
echo "You're a release analyst. I have a file where I need:"
echo "- Standardized date formats (ISO 8601: YYYY-MM-DD)"
echo "- Removal or commenting of placeholder values"
echo "- Suggestions on ambiguous hardcoded items or TODOs"
echo ""
echo "Below is a sample from the file:"
echo ""
echo "--- DATES FOUND ---"
echo "$DATES"
echo
echo "--- HARDCODED / TAGS / TODOS ---"
echo "$HARDCODED"
echo
echo "--- PLACEHOLDER PATTERNS ---"
echo "$PLACEHOLDERS"
echo
echo "Return a cleaned diff or proposed changelog corrections."
echo '"""'
echo
echo "📌 Paste the file content below this prompt in your LLM."
echo
echo "✅ When ready, type ENTER here to proceed with manual edits."
read
echo "🛠️  You can now begin editing or running cleanup scripts manually."
