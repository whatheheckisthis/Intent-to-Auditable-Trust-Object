#!/usr/bin/env bash
set -euo pipefail

mkdir -p /opt/netflow

if [[ ! -r /sys/kernel/btf/vmlinux ]]; then
  echo "[startup] missing /sys/kernel/btf/vmlinux for CO-RE" >&2
  exit 1
fi

bpftool btf dump file /sys/kernel/btf/vmlinux format c > /opt/netflow/vmlinux.h

clang -O2 -g -target bpf -D__TARGET_ARCH_x86 \
  -I/opt/netflow \
  -c /opt/netflow/netflow_filter.bpf.c \
  -o /opt/netflow/netflow_filter.bpf.o

exec /usr/local/bin/netflow-sidecar
