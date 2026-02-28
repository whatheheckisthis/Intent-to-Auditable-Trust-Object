#!/usr/bin/env bash
set -euo pipefail

if command -v git >/dev/null 2>&1 && command -v git-gui >/dev/null 2>&1; then
  echo "git + git-gui already available"
  git --version || true
  git-gui --version || true
  exit 0
fi

echo "Installing git + git-gui..."
if ! command -v apt-get >/dev/null 2>&1; then
  echo "apt-get not available on this system"
  exit 1
fi

set +e
apt-get update
apt_status=$?
set -e
if [ "$apt_status" -ne 0 ]; then
  echo "apt-get update failed (often proxy/network restriction)."
  echo "Current proxy env (if set):"
  env | rg -i 'proxy' || true
  exit "$apt_status"
fi

apt-get install -y git git-gui

if command -v git-gui >/dev/null 2>&1; then
  echo "git-gui installed: $(command -v git-gui)"
else
  echo "git-gui installation appears incomplete"
  exit 1
fi

# Optional convenience alias to launch git-gui quickly.
GIT_GUI_ALIAS='alias ggui="git gui"'
if ! grep -q 'alias ggui=' "$HOME/.bashrc" 2>/dev/null; then
  echo "$GIT_GUI_ALIAS" >> "$HOME/.bashrc"
  echo "Added alias to ~/.bashrc: $GIT_GUI_ALIAS"
fi

echo "Done. Run: git gui  (or ggui in a new shell)"
