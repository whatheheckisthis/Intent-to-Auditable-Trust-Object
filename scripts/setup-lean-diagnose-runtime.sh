#!/usr/bin/env bash
set -euo pipefail

# Installs/configures Docker or Podman for lean-diagnose workflows.
#
# Usage:
#   ./scripts/setup-lean-diagnose-runtime.sh
#   ./scripts/setup-lean-diagnose-runtime.sh --check
#
# Optional environment overrides:
#   APT_MIRROR=http://mirror.example/ubuntu
#   APT_SECURITY_MIRROR=http://security.ubuntu.com/ubuntu
#   APT_FORCE_HTTPS=1
#   APT_PROXY=http://proxy:8080
#   APT_DISABLE_THIRD_PARTY=1
#   DNS_NAMESERVERS="1.1.1.1 8.8.8.8"
#   INSTALL_RETRIES=2

CHECK_ONLY=0
if [[ "${1:-}" == "--check" ]]; then
  CHECK_ONLY=1
fi

INSTALL_RETRIES="${INSTALL_RETRIES:-2}"

have_cmd() { command -v "$1" >/dev/null 2>&1; }

print_runtime_state() {
  echo "[state] docker: $(command -v docker || echo missing)"
  echo "[state] podman: $(command -v podman || echo missing)"
  echo "[state] docker compose: $(docker compose version 2>/dev/null || echo missing)"
  echo "[state] podman-compose: $(command -v podman-compose || echo missing)"
}

configure_dns_resolver() {
  local resolv="/etc/resolv.conf"
  local nameservers="${DNS_NAMESERVERS:-}"

  if [[ -n "$nameservers" ]]; then
    echo "[dns] writing explicit nameservers to $resolv"
    {
      echo "# managed by setup-lean-diagnose-runtime.sh"
      for ns in $nameservers; do
        echo "nameserver $ns"
      done
    } >"$resolv"
    return
  fi

  if grep -Eq '^nameserver[[:space:]]+[0-9a-fA-F:.]+' "$resolv" 2>/dev/null; then
    echo "[dns] nameserver entry detected in $resolv"
    return
  fi

  echo "[dns] no nameserver configured; applying fallback resolvers"
  cat >"$resolv" <<RESOLV
# managed by setup-lean-diagnose-runtime.sh fallback
nameserver 1.1.1.1
nameserver 8.8.8.8
RESOLV
}

validate_dns_resolution() {
  local host="${DNS_TEST_HOST:-archive.ubuntu.com}"
  if have_cmd getent && getent hosts "$host" >/dev/null 2>&1; then
    echo "[dns] resolved $host via getent"
    return 0
  fi

  if have_cmd nslookup && nslookup "$host" >/dev/null 2>&1; then
    echo "[dns] resolved $host via nslookup"
    return 0
  fi

  echo "[dns] WARNING: unable to resolve $host; apt may fail"
  return 1
}

configure_apt_proxy() {
  if [[ -n "${APT_PROXY:-}" ]]; then
    echo "[apt] configuring proxy: $APT_PROXY"
    cat >/etc/apt/apt.conf.d/99lean-diagnose-proxy <<APTCONF
Acquire::http::Proxy "${APT_PROXY}";
Acquire::https::Proxy "${APT_PROXY}";
APTCONF
  fi
}

configure_apt_mirror() {
  local src=/etc/apt/sources.list

  if [[ "${APT_DISABLE_THIRD_PARTY:-1}" == "1" ]]; then
    echo "[apt] disabling third-party sources (best effort)"
    if [[ -d /etc/apt/sources.list.d ]]; then
      find /etc/apt/sources.list.d -type f \
        \( -name '*llvm*' -o -name '*mise*' -o -name '*.list' -o -name '*.sources' \) \
        -print0 2>/dev/null |
        xargs -0 -r sed -i 's|^deb |# deb |g' || true
    fi
  fi

  if [[ -f "$src" ]]; then
    if [[ -n "${APT_MIRROR:-}" ]]; then
      sed -i "s|http://archive.ubuntu.com/ubuntu|${APT_MIRROR}|g" "$src"
    fi
    if [[ -n "${APT_SECURITY_MIRROR:-}" ]]; then
      sed -i "s|http://security.ubuntu.com/ubuntu|${APT_SECURITY_MIRROR}|g" "$src"
    fi
    if [[ "${APT_FORCE_HTTPS:-0}" == "1" ]]; then
      sed -i 's|http://archive.ubuntu.com/ubuntu|https://archive.ubuntu.com/ubuntu|g' "$src"
      sed -i 's|http://security.ubuntu.com/ubuntu|https://security.ubuntu.com/ubuntu|g' "$src"
    fi
  fi

  if [[ -d /etc/apt/sources.list.d ]]; then
    for f in /etc/apt/sources.list.d/*.sources /etc/apt/sources.list.d/*.list; do
      [[ -e "$f" ]] || continue
      if [[ -n "${APT_MIRROR:-}" ]]; then
        sed -i "s|http://archive.ubuntu.com/ubuntu|${APT_MIRROR}|g" "$f"
      fi
      if [[ -n "${APT_SECURITY_MIRROR:-}" ]]; then
        sed -i "s|http://security.ubuntu.com/ubuntu|${APT_SECURITY_MIRROR}|g" "$f"
      fi
      if [[ "${APT_FORCE_HTTPS:-0}" == "1" ]]; then
        sed -i 's|http://archive.ubuntu.com/ubuntu|https://archive.ubuntu.com/ubuntu|g' "$f"
        sed -i 's|http://security.ubuntu.com/ubuntu|https://security.ubuntu.com/ubuntu|g' "$f"
      fi
    done
  fi
}

retry() {
  local attempt=1
  local max=$((INSTALL_RETRIES + 1))
  until "$@"; do
    if (( attempt >= max )); then
      echo "[retry] command failed after $attempt attempt(s): $*"
      return 1
    fi
    attempt=$((attempt + 1))
    echo "[retry] retrying ($attempt/$max): $*"
    sleep 1
  done
}

install_with_apt() {
  echo "[install] using apt-get"
  export DEBIAN_FRONTEND=noninteractive
  configure_dns_resolver
  validate_dns_resolution || true
  configure_apt_proxy
  configure_apt_mirror
  retry apt-get update -y
  retry apt-get install -y podman podman-compose || \
    retry apt-get install -y docker.io docker-compose-plugin
}

install_with_apk() {
  echo "[install] using apk"
  configure_dns_resolver
  validate_dns_resolution || true
  retry apk update
  retry apk add --no-cache podman podman-compose || \
    retry apk add --no-cache docker-cli docker-compose
}

install_with_dnf() {
  echo "[install] using dnf"
  configure_dns_resolver
  validate_dns_resolution || true
  retry dnf install -y podman podman-compose || \
    retry dnf install -y docker docker-compose-plugin
}

install_with_yum() {
  echo "[install] using yum"
  configure_dns_resolver
  validate_dns_resolution || true
  retry yum install -y podman podman-compose || \
    retry yum install -y docker docker-compose-plugin
}

install_with_pacman() {
  echo "[install] using pacman"
  configure_dns_resolver
  validate_dns_resolution || true
  retry pacman -Sy --noconfirm podman podman-compose || \
    retry pacman -Sy --noconfirm docker docker-compose
}

if have_cmd docker || have_cmd podman; then
  echo "[setup] container runtime already present"
  print_runtime_state
  exit 0
fi

if [[ "$CHECK_ONLY" -eq 1 ]]; then
  echo "[check] no runtime installed; run without --check to attempt installation"
  print_runtime_state
  exit 1
fi

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
  echo "[install] unsupported package manager; install docker or podman manually"
  exit 2
fi

if have_cmd docker; then
  echo "[configure] enabling docker service (best effort)"
  systemctl enable --now docker 2>/dev/null || service docker start 2>/dev/null || true
fi

if have_cmd podman; then
  echo "[configure] podman detected (no daemon required)"
fi

if have_cmd docker || have_cmd podman; then
  echo "[setup] runtime installation/configuration complete"
  print_runtime_state
  exit 0
fi

echo "[setup] installation attempted but runtime is still missing"
print_runtime_state
exit 3
