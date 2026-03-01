#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 <worker_id> <world:rme|normal> [deps_pipe_separated]" >&2
  exit 1
fi

worker_id="$1"
world="$2"
deps="${3:-}"

case "$world" in
  rme|normal) ;;
  *)
    echo "Invalid world '$world'. Expected 'rme' or 'normal'." >&2
    exit 1
    ;;
esac

if [[ ! "$worker_id" =~ ^[a-zA-Z0-9._-]+$ ]]; then
  echo "Invalid worker_id '$worker_id'. Use letters, numbers, dot, underscore, or dash." >&2
  exit 1
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
worker_dir="$repo_root/workers/new/$worker_id"
worker_file="$worker_dir/worker.json"

mkdir -p "$worker_dir"

cat > "$worker_file" <<JSON
{
  "id": "$worker_id",
  "world": "$world",
  "deps": "${deps}",
  "architecture": "IATO-V7",
  "source": "scripts/setup_worker.sh"
}
JSON

echo "Created worker scaffold: $worker_file"
