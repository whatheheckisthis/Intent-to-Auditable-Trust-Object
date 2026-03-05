#!/usr/bin/env bash
set -euo pipefail

repo_root="$(pwd)"
chmod +x "$repo_root/bin/iato"
mkdir -p "$HOME/bin"
ln -sfn "$repo_root/bin/iato" "$HOME/bin/iato"

echo "CLI setup complete."
echo "Now you can run: iato scan --config config.local.toml"
echo 'If needed: export PATH="$HOME/bin:$PATH"'
