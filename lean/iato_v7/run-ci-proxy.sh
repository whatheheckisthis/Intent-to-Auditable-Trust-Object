#!/usr/bin/env sh
set -eu

SCAFFOLD_FILE="${SCAFFOLD_FILE:-/config/k8s-build-job.yaml}"
WORKDIR="${WORKDIR:-/workspace}"

if [ ! -f "$SCAFFOLD_FILE" ]; then
  echo "Scaffold not found: $SCAFFOLD_FILE" >&2
  exit 1
fi

CMD="$(awk '
  function indent_len(s,   n,i,c) {
    n=0
    for (i=1; i<=length(s); i++) {
      c=substr(s,i,1)
      if (c == " ") n++
      else break
    }
    return n
  }
  {
    line=$0
    indent=indent_len(line)

    if (!in_args && line ~ /^[[:space:]]*args:[[:space:]]*$/) {
      in_args=1
      args_indent=indent
      next
    }

    if (in_args && !capture && line ~ /^[[:space:]]*-[[:space:]]*>-[[:space:]]*$/) {
      capture=1
      cmd_indent=indent + 2
      next
    }

    if (capture) {
      if (line ~ /^[[:space:]]*$/) next
      if (indent < cmd_indent) {
        capture=0
        in_args=0
      } else {
        sub(/^[[:space:]]+/, "", line)
        cmd = cmd line " "
      }
    }
  }
  END { gsub(/[[:space:]]+$/, "", cmd); print cmd }
' "$SCAFFOLD_FILE")"

if [ -z "$CMD" ]; then
  echo "Unable to parse scaffold args from $SCAFFOLD_FILE" >&2
  exit 1
fi

echo "[ci-proxy] cd $WORKDIR"
cd "$WORKDIR"
echo "[ci-proxy] exec: $CMD"
exec /bin/sh -lc "$CMD"
