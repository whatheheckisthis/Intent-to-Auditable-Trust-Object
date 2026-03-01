#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <source.c> <function_name> <arch> [compiler]" >&2
  echo "  arch: arm64 | x86_64" >&2
  exit 2
fi

SRC="$1"
FUNC="$2"
ARCH="$3"
CC_BIN="${4:-cc}"

WORKDIR="$(mktemp -d)"
OBJ="$WORKDIR/out.o"
DIS="$WORKDIR/disasm.txt"
HEX="$WORKDIR/hex_and_mnemonic.txt"

trap 'rm -rf "$WORKDIR"' EXIT

"$CC_BIN" -O2 -c "$SRC" -o "$OBJ"

if command -v llvm-objdump >/dev/null 2>&1; then
  llvm-objdump -d "$OBJ" > "$DIS"
else
  objdump -d "$OBJ" > "$DIS"
fi

awk '/^[[:space:]]*[0-9a-f]+:/{print}' "$DIS" > "$HEX"

echo "== Disassembly (hex + mnemonic) for $FUNC =="
awk -v f="$FUNC" '
  $0 ~ "<"f">:" {infn=1; print; next}
  infn && /^[[:space:]]*[0-9a-f]+:/ {print; next}
  infn && !/^[[:space:]]*[0-9a-f]+:/ {infn=0}
' "$DIS"

if [[ "$ARCH" == "arm64" ]]; then
  # Disallow conditional branches/selects that can reintroduce data-dependent control.
  BAD_REGEX='(^|[^[:alnum:]_])(b\.[a-z]+|cbz|cbnz|tbz|tbnz|csel|csinc|csinv|csneg)($|[^[:alnum:]_])'
elif [[ "$ARCH" == "x86_64" ]]; then
  BAD_REGEX='(^|[^[:alnum:]_])(j[a-z]+|cmov[a-z]+)($|[^[:alnum:]_])'
else
  echo "Unsupported arch: $ARCH" >&2
  exit 2
fi

if awk -v f="$FUNC" -v bad="$BAD_REGEX" '
  BEGIN { infn=0; badfound=0 }
  $0 ~ "<"f">:" { infn=1; next }
  infn && !/^[[:space:]]*[0-9a-f]+:/ { infn=0 }
  infn {
    line=tolower($0)
    if (line ~ bad) {
      print "forbidden instruction in " f ": " $0
      badfound=1
    }
  }
  END { exit badfound ? 1 : 0 }
' "$DIS"; then
  echo "PASS: No forbidden conditional instructions detected in $FUNC"
else
  echo "FAIL: Forbidden conditional instruction(s) detected in $FUNC" >&2
  exit 1
fi
