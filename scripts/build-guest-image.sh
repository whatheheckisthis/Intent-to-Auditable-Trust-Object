#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build/guest"
BASE_IMG="${BUILD_DIR}/base.img"
RAW_IMG="${BUILD_DIR}/guest-raw.img"
FINAL_IMG="${BUILD_DIR}/guest.img"
CACHE_DIR="${BUILD_DIR}/apt-cache"
WHEEL_DIR="${BUILD_DIR}/wheels"

BASE_URL="https://cloud-images.ubuntu.com/minimal/releases/24.04/release"
BASE_FILE="ubuntu-24.04-minimal-cloudimg-arm64.img"
SHA_FILE="SHA256SUMS"

mkdir -p "${BUILD_DIR}" "${CACHE_DIR}" "${WHEEL_DIR}" "${ROOT_DIR}/build/el2"

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "[build-guest][error] missing required command: $1" >&2
    exit 1
  fi
}

need_cmd curl
need_cmd sha256sum
need_cmd qemu-img
need_cmd virt-customize

sha_expected="$(curl -fsSL "${BASE_URL}/${SHA_FILE}" | awk -v f="${BASE_FILE}" '$2==f {print $1}')"
if [[ -z "${sha_expected}" ]]; then
  echo "[build-guest][error] unable to resolve expected checksum for ${BASE_FILE}" >&2
  exit 1
fi

if [[ -f "${BASE_IMG}" ]]; then
  sha_actual="$(sha256sum "${BASE_IMG}" | awk '{print $1}')"
  if [[ "${sha_actual}" != "${sha_expected}" ]]; then
    echo "[build-guest] cached base image checksum mismatch; refreshing"
    rm -f "${BASE_IMG}"
  fi
fi

if [[ ! -f "${BASE_IMG}" ]]; then
  curl -fL "${BASE_URL}/${BASE_FILE}" -o "${BASE_IMG}"
fi

sha_actual="$(sha256sum "${BASE_IMG}" | awk '{print $1}')"
if [[ "${sha_actual}" != "${sha_expected}" ]]; then
  echo "[build-guest][error] checksum verification failed for base image" >&2
  exit 1
fi

if [[ ! -d "${WHEEL_DIR}" ]] || ! find "${WHEEL_DIR}" -maxdepth 1 -name '*.whl' | grep -q .; then
  echo "[build-guest][error] wheel cache not found; run make fetch-wheels first" >&2
  exit 1
fi

rm -f "${RAW_IMG}" "${FINAL_IMG}"
qemu-img convert -f qcow2 -O raw "${BASE_IMG}" "${RAW_IMG}"

virt-customize -a "${RAW_IMG}" \
  --run-command 'mkdir -p /etc/apt/apt.conf.d && printf "Dir::Cache::archives \"/var/cache/apt/archives\";\n" > /etc/apt/apt.conf.d/99iato-cache' \
  --copy-in "${CACHE_DIR}/:/var/cache/apt/archives" \
  --run-command 'apt-get update && apt-get install -y python3 python3-pip tpm2-tools tpm2-pytss' \
  --mkdir /opt/iato \
  --copy-in "${ROOT_DIR}/src:/opt/iato" \
  --copy-in "${ROOT_DIR}/tests:/opt/iato" \
  --copy-in "${ROOT_DIR}/pyproject.toml:/opt/iato" \
  --mkdir /opt/iato/scripts \
  --copy-in "${ROOT_DIR}/scripts/activate-dotnet-offline.sh:/opt/iato/scripts" \
  --copy-in "${WHEEL_DIR}:/opt/iato/build/guest" \
  --run-command 'python3 -m pip install --no-index --find-links /opt/iato/build/guest/wheels cryptography==42.0.5 pytest==8.1.0 pytest-cov==5.0.0' \
  --run-command 'cat > /etc/rc.local <<"RCLOCAL"\n#!/bin/sh -e\necho "[guest-ready]" > /dev/hvc0\nexit 0\nRCLOCAL\nchmod +x /etc/rc.local'

qemu-img convert -f raw -O qcow2 "${RAW_IMG}" "${FINAL_IMG}"
qemu-img resize "${FINAL_IMG}" 8G >/dev/null

size_mb="$(du -m "${FINAL_IMG}" | awk '{print $1}')"
echo "[build-guest] OK: build/guest/guest.img (${size_mb}MB)"
