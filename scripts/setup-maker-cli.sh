#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$HOME/.local/bin"
ln -sfn "$repo_root/bin/iato" "$HOME/.local/bin/iato"

echo "Maker fix installed: ~/.local/bin/iato -> $repo_root/bin/iato"
echo 'If needed, add this once: export PATH="$HOME/.local/bin:$PATH"'
