#!/usr/bin/env bash
set -euo pipefail

# Installs Apache and Nginx from distro package stores.
# Adds apt-specific mitigation for mirror/proxy/DNS-restricted environments.
#
# Usage:
#   ./scripts/install-web-proxies.sh
#
# Optional environment overrides (apt path):
#   APT_PROXY=http://proxy.internal:8080
#   APT_MIRROR=http://mirror.internal/ubuntu
#   APT_SECURITY_MIRROR=http://mirror.internal/ubuntu-security
#   APT_FORCE_HTTPS=1
#   APT_DISABLE_THIRD_PARTY=1
#   DNS_NAMESERVERS="1.1.1.1 8.8.8.8"

have_cmd() { command -v "$1" >/dev/null 2>&1; }

configure_dns() {
  local resolv="/etc/resolv.conf"
  if [[ -n "${DNS_NAMESERVERS:-}" ]]; then
    echo "[dns] configuring resolvers: ${DNS_NAMESERVERS}"
    {
      echo "# managed by install-web-proxies.sh"
      for ns in $DNS_NAMESERVERS; do
        echo "nameserver $ns"
      done
    } >"$resolv"
  fi
}

configure_apt_proxy_and_mirrors() {
  if [[ -n "${APT_PROXY:-}" ]]; then
    echo "[apt] configuring proxy: $APT_PROXY"
    cat >/etc/apt/apt.conf.d/99web-proxy-installer <<APTCONF
Acquire::http::Proxy "${APT_PROXY}";
Acquire::https::Proxy "${APT_PROXY}";
APTCONF
  fi

  if [[ "${APT_DISABLE_THIRD_PARTY:-1}" == "1" && -d /etc/apt/sources.list.d ]]; then
    echo "[apt] disabling third-party sources (best effort)"
    find /etc/apt/sources.list.d -type f \( -name '*.list' -o -name '*.sources' \) \
      -print0 2>/dev/null | xargs -0 -r sed -i 's|^deb |# deb |g' || true
  fi

  for f in /etc/apt/sources.list \
  /etc/apt/sources.list.d/*.list \
  /etc/apt/sources.list.d/*.sources; do
    [[ -e "$f" ]] || continue
    if [[ -n "${APT_MIRROR:-}" ]]; then
      sed -i "s|http://archive.ubuntu.com/ubuntu|${APT_MIRROR}|g" "$f" || true
    fi
    if [[ -n "${APT_SECURITY_MIRROR:-}" ]]; then
      sed -i "s|http://security.ubuntu.com/ubuntu|${APT_SECURITY_MIRROR}|g" "$f" || true
    fi
    if [[ "${APT_FORCE_HTTPS:-0}" == "1" ]]; then
      sed -i 's|http://archive.ubuntu.com/ubuntu|https://archive.ubuntu.com/ubuntu|g' "$f" || true
      sed -i 's|http://security.ubuntu.com/ubuntu|https://security.ubuntu.com/ubuntu|g' "$f" || true
    fi
  done
}

install_with_apt() {
  export DEBIAN_FRONTEND=noninteractive
  configure_dns
  configure_apt_proxy_and_mirrors

  set +e
  local out
  out=$(apt-get update -y 2>&1)
  local rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "$out"
    if grep -q "403" <<<"$out"; then
      echo "[apt] detected HTTP 403 from upstream mirror/proxy"
      echo "[apt] mitigation: provide internal mirror/proxy via env vars:"
      echo "       APT_PROXY, APT_MIRROR, APT_SECURITY_MIRROR, APT_FORCE_HTTPS=1"
    fi
    return $rc
  fi

  apt-get install -y apache2 nginx
}

install_with_apk() {
  apk update
  apk add --no-cache apache2 nginx
}

install_with_dnf() {
  dnf install -y httpd nginx
}

install_with_yum() {
  yum install -y httpd nginx
}

install_with_pacman() {
  pacman -Sy --noconfirm apache nginx
}

if have_cmd apt-get; then
  install_with_apt
elif have_cmd apk; then
  install_with_apk
elif have_cmd dnf; then
  install_with_dnf
elif have_cmd yum; then
  install_with_yum
elif have_cmd pacman; then
  install_with_pacman
else
  echo "Unsupported package manager"
  exit 2
fi

echo "Installed apache + nginx successfully"
