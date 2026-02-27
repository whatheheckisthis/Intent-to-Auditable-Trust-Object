#!/usr/bin/env bash
set -euo pipefail

# Starts a local apt reverse proxy with nginx or apache (whichever exists).
#
# Usage:
#   ./scripts/setup-apt-reverse-proxy.sh
#   ./scripts/setup-apt-reverse-proxy.sh nginx
#   ./scripts/setup-apt-reverse-proxy.sh apache
#
# Then run:
#   export APT_PROXY=http://127.0.0.1:3142
#   export APT_MIRROR=http://127.0.0.1:3142/ubuntu
#   export APT_SECURITY_MIRROR=http://127.0.0.1:3142/ubuntu-security
#   ./scripts/setup-lean-diagnose-runtime.sh

ENGINE="${1:-auto}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

start_nginx() {
  local cfg="$ROOT_DIR/proxy/nginx/apt-proxy.conf"
  echo "[proxy] starting nginx with $cfg"
  nginx -c "$cfg"
}

start_apache() {
  local cfg="$ROOT_DIR/proxy/apache/apt-proxy.conf"
  echo "[proxy] starting apache with $cfg"
  if command -v apache2ctl >/dev/null 2>&1; then
    apache2ctl -f "$cfg" -k start
  else
    httpd -f "$cfg" -k start
  fi
}

if [[ "$ENGINE" == "nginx" ]]; then
  command -v nginx >/dev/null 2>&1 || {
    echo "[proxy] nginx not installed"
    exit 2
  }
  start_nginx
elif [[ "$ENGINE" == "apache" ]]; then
  command -v apache2ctl >/dev/null 2>&1 || command -v httpd >/dev/null 2>&1 || {
    echo "[proxy] apache not installed"
    exit 2
  }
  start_apache
else
  if command -v nginx >/dev/null 2>&1; then
    start_nginx
  elif command -v apache2ctl >/dev/null 2>&1 || command -v httpd >/dev/null 2>&1; then
    start_apache
  else
    echo "[proxy] neither nginx nor apache is installed"
    exit 2
  fi
fi

echo "[proxy] ready on http://127.0.0.1:3142"
echo "[proxy] export APT_PROXY=http://127.0.0.1:3142"
echo "[proxy] export APT_MIRROR=http://127.0.0.1:3142/ubuntu"
echo "[proxy] export APT_SECURITY_MIRROR=http://127.0.0.1:3142/ubuntu-security"
