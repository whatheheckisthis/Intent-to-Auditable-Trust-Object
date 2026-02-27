#!/usr/bin/env bash
set -euo pipefail

# Wrapper that applies local/proxy mirror settings then installs Apache+Nginx.
#
# Intended replacement for failing direct commands:
#   sudo apt-get update -y
#   sudo apt-get install -y apache2 nginx
#
# Usage:
#   ./scripts/fix-apt-403-and-install-web-proxies.sh
#
# If local reverse proxy is running (scripts/setup-apt-reverse-proxy.sh),
# this script defaults to it.

export APT_PROXY="${APT_PROXY:-http://127.0.0.1:3142}"
export APT_MIRROR="${APT_MIRROR:-http://127.0.0.1:3142/ubuntu}"
export APT_SECURITY_MIRROR="${APT_SECURITY_MIRROR:-http://127.0.0.1:3142/ubuntu-security}"
export APT_FORCE_HTTPS="${APT_FORCE_HTTPS:-0}"
export APT_DISABLE_THIRD_PARTY="${APT_DISABLE_THIRD_PARTY:-1}"

exec "$(dirname "$0")/install-web-proxies.sh"
