#!/usr/bin/env bash
set -euo pipefail

# Preflight checker for PR conflict/debug workflows.
# Usage:
#   scripts/pr_conflict_preflight.sh [files...]

if ! command -v git >/dev/null 2>&1; then
  echo "ERROR: git is not installed" >&2
  exit 1
fi

if [[ $# -eq 0 ]]; then
  files=(
    "IATO_V7/IATO/V7/RMEModel.lean"
    "IATO_V7/Test/RME.lean"
    "scripts/generate_rme_compliance_artifacts.py"
    "formal/README.md"
    "artifacts/compliance/armv9_rme_evidence.json"
    "artifacts/compliance/armv9_rme_evidence.md"
  )
else
  files=("$@")
fi

echo "== Git branch =="
git rev-parse --abbrev-ref HEAD

echo "== Unmerged files =="
if git diff --name-only --diff-filter=U | tee /tmp/iato-unmerged.txt | grep -q .; then
  echo "ERROR: unmerged files detected" >&2
  exit 2
else
  echo "none"
fi

echo "== Conflict markers =="
conflict_found=0
for f in "${files[@]}"; do
  if [[ -f "$f" ]]; then
    if rg -n "^(<<<<<<<|=======|>>>>>>>)" "$f"; then
      conflict_found=1
    fi
  fi
done
if [[ $conflict_found -eq 1 ]]; then
  echo "ERROR: conflict markers found" >&2
  exit 3
fi

echo "none"

echo "== Compliance generator checks =="
python3 -m py_compile scripts/generate_rme_compliance_artifacts.py
python3 scripts/generate_rme_compliance_artifacts.py
python3 -m json.tool artifacts/compliance/armv9_rme_evidence.json >/dev/null

echo "== Lean test availability =="
if command -v lake >/dev/null 2>&1; then
  (cd IATO_V7 && lake test)
else
  echo "WARNING: lake not installed; skipped Lean tests" >&2
fi

echo "Preflight complete"
