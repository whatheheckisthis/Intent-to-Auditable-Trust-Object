#!/usr/bin/env sh
set -eu

TARGET="docs/notebooks/lean/IATO_V7_GPTMLE.lean"
SCAFFOLD="docs/notebooks/lean/IATO_V7_Scaffold.lean"

if [ ! -f "$TARGET" ]; then
  echo "ERROR: missing $TARGET"
  exit 1
fi

echo "[virtual-lean] target: $TARGET"

if command -v lean >/dev/null 2>&1; then
  echo "[toolchain] lean found: $(command -v lean)"
else
  echo "[toolchain] lean not found (virtual mode only)"
fi

if command -v lake >/dev/null 2>&1; then
  echo "[toolchain] lake found: $(command -v lake)"
else
  echo "[toolchain] lake not found (virtual mode only)"
fi

awk 'length>100{print "[line-length] " NR ":" length}' "$TARGET"

SORRY_COUNT=$(rg -n "\bsorry\b" "$TARGET" | wc -l | tr -d ' ')
echo "[virtual-lean] sorry count: $SORRY_COUNT"

if ! rg -n "deriving DecidableEq" "$SCAFFOLD" >/dev/null 2>&1; then
  echo "[virtual-lean] WARNING: Granule may miss DecidableEq"
else
  echo "[virtual-lean] Granule DecidableEq derivation detected"
fi

echo "[virtual-lean] synthetic diagnostics complete"
