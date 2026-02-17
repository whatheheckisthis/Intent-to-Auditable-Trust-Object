# Intent-to-Auditable Trust Object: XDP → BPF Maps → Enclave SDN

This repository now includes a complete, hardware-software co-designed monitoring stack that bridges:

- **Layer 1 (XDP hook):** high-speed packet parsing + timing capture in kernel space.
- **Layer 2 (BPF Maps):** `flow_map` in `BPF_MAP_TYPE_PERCPU_HASH` sized for **1M concurrent flows**.
- **Layer 3 (Enclave SDN):** Prometheus, Grafana, and sidecar all bound to the Enclave service namespace.

---

## 1) Components Delivered

### XDP kernel module (`ebpf/xdp_audit.c`)

- Exposes XDP program section `netflow_filter`.
- Parses Ethernet/IPv4/TCP/UDP and derives 5-tuple flow keys.
- Tracks per-flow packet/byte counters and first/last-seen timestamps in:
  - `flow_map` (`BPF_MAP_TYPE_PERCPU_HASH`, `max_entries=1048576`).
- Uses `bpf_ktime_get_ns()` for nanosecond packet timing deltas.
- Emits audit alerts through `events` (`BPF_MAP_TYPE_RINGBUF`) when:
  - packet-rate spike thresholds are exceeded,
  - ultra-low inter-packet timing deltas suggest potential side-channel leakage.

### Go sidecar bridge (`ebpf/netflow-sidecar/main.go`)

- Uses **libbpf via `libbpfgo`** to:
  - load CO-RE object `/opt/netflow/xdp_audit.bpf.o`,
  - attach `netflow_filter` to the configured NIC.
- Aggregates per-CPU flow counters from `flow_map` into global snapshots.
- Publishes Prometheus metrics at `/metrics` (default `:9400`).
- Subscribes to the ring buffer (`events`) and logs audit messages as:
  - `Trust Violation reason=...`

### Enclave-wrapped compose topology (`docker-compose.yml`)

- `xdp-auditor`, `prometheus`, and `grafana` run with:
  - `network_mode: "service:enclave"`.
- BPF sidecar runs with:
  - `privileged: true`,
  - `/sys/fs/bpf` mount,
  - `/lib/modules` mount,
  - `/sys/kernel/btf` mount.
- App plane (`drupal`, `db`, `apache`) remains isolated on `app_net` and does not need awareness of the eBPF monitor.

---

## 2) WSL2 Quick Start

1. Copy env template:

```bash
cp .env.example .env
```

2. Set required secrets in `.env`:

- `ENCLAVE_ENROLMENT_KEY`
- `DRUPAL_DB_PASSWORD`
- `MARIADB_ROOT_PASSWORD`
- `GRAFANA_ADMIN_PASSWORD`

3. Optional XDP tuning:

```bash
XDP_INTERFACE=eth0
XDP_MODE=drv
NETFLOW_POLL_INTERVAL=10s
NETFLOW_TOPK=50
```

4. Build and start:

```bash
docker compose up -d --build
```

---

## 3) Verify the XDP-to-Enclave Pipeline

### A. Confirm XDP program is attached

```bash
docker logs xdp-auditor --tail=100
```

Expected log includes:

- `netflow_filter attached via libbpf on <iface>`

### B. Confirm ringbuf trust-violation stream

```bash
docker logs -f xdp-auditor
```

Look for entries similar to:

- `Trust Violation reason=1 ...`
- `Trust Violation reason=2 ...`

### C. Confirm Prometheus sees flow metrics

```bash
docker exec enclave_agent wget -qO- http://localhost:9400/metrics | head -40
```

Metrics include:

- `xdp_netflow_active_flows`
- `xdp_netflow_packets_total_snapshot`
- `xdp_netflow_flow_packets`

### D. Confirm dashboard endpoints via Enclave virtual IP / name

Open in your browser:

- `https://grafana.enclave`

If your Enclave setup assigns a specific virtual IP for Grafana, use:

- `https://<enclave-virtual-ip>:443`

In Grafana, query:

- `xdp_netflow_active_flows`
- `xdp_netflow_flow_packets`
- `xdp_netflow_flow_bytes`

for real-time, line-rate flow telemetry and timing audit visibility.

---

## 4) Delivered Files

- `ebpf/xdp_audit.c`
- `ebpf/netflow-sidecar/main.go`
- `docker-compose.yml`
- `README.md`
