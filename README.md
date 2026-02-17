# Auditable Trust Object: Enclave + eBPF/XDP Monitoring Stack

This repository ships a secure and auditable monitoring architecture that combines:

- **Line-rate kernel telemetry** from a CO-RE XDP/eBPF netflow filter (`ebpf/netflow_filter.bpf.c`),
- **A Go-based enclave sidecar** that reads an eBPF per-CPU flow map and exports Prometheus metrics,
- **A private overlay network backbone** for observability and dashboard access,
- **An isolated app plane** (`Nginx -> Drupal -> MariaDB`) on `app_net`.

## 1) Architecture

### Software-Verilog Layer (Kernel/NIC path)

- `ebpf/netflow_filter.bpf.c` attaches at the XDP hook and is intended for **`XDP_DRV`** (driver mode) and optionally **`XDP_HW`** where NIC offload is supported.
- The program parses IPv4 TCP/UDP headers and implements `netflow_filter` by aggregating packet + byte counters per 5-tuple flow.
- Flow state is stored in `flow_map` using `BPF_MAP_TYPE_PERCPU_HASH` so each CPU updates local counters without cross-core lock contention.
- The BPF object is built with BTF/CO-RE (`vmlinux.h` generated from `/sys/kernel/btf/vmlinux`) for portability across compatible kernels.

### Bridge Layer (Userspace sidecar)

- `ebpf/netflow-sidecar/main.go` loads and attaches the XDP program, then polls `flow_map`.
- The sidecar exports `/metrics` on port `9400` over the enclave vNET namespace.
- Exported metrics include:
  - `xdp_netflow_active_flows`
  - `xdp_netflow_packets_total_snapshot`
  - `xdp_netflow_bytes_total_snapshot`
  - `xdp_netflow_flow_packets` (top-N)
  - `xdp_netflow_flow_bytes` (top-N)
- Container runs with:
  - `network_mode: "service:enclave"`
  - `privileged: true`
  - host mount for `/sys/kernel/btf`.

### Hybrid Compose Topology

- **Enclave service** provides the secure networking backbone and `.enclave` name resolution.
- **Prometheus + Grafana** run in the enclave network namespace and scrape the XDP sidecar privately.
- **App stack** (`drupal`, `apache`, `db`) remains isolated on `app_net`.
- **Gateway Nginx** is attached to the enclave namespace and exposes TLS on host port `443`.

## 2) Key Files

- `ebpf/netflow_filter.bpf.c`
- `ebpf/netflow-sidecar/main.go`
- `ebpf/netflow-sidecar/entrypoint.sh`
- `ebpf/netflow-sidecar/Dockerfile`
- `docker-compose.yml`
- `prometheus.yml`

## 3) WSL2 Quick Start

1. **Prerequisites**

   - Docker Desktop with WSL2 backend enabled.
   - Linux kernel with eBPF/XDP support and `/sys/kernel/btf/vmlinux` exposed.
   - Enclave enrolment key.

2. **Prepare environment**

   ```bash
   cp .env.example .env
   ```

   Set at minimum:

   - `ENCLAVE_ENROLMENT_KEY`
   - `DRUPAL_DB_PASSWORD`
   - `MARIADB_ROOT_PASSWORD`
   - `GRAFANA_ADMIN_PASSWORD`

   Optional XDP tuning:

   - `XDP_INTERFACE=eth0`
   - `XDP_MODE=drv` (`drv`, `hw`, or `skb`)
   - `NETFLOW_POLL_INTERVAL=10s`
   - `NETFLOW_TOPK=50`

3. **Create TLS certificate**

   ```bash
   mkdir -p certs
   openssl req -x509 -nodes -newkey rsa:2048 -days 365 \
     -keyout certs/tls.key \
     -out certs/tls.crt \
     -subj "/CN=grafana.enclave"
   ```

4. **Launch stack**

   ```bash
   docker compose up -d --build
   ```

## 4) Verification

### A. Verify XDP hook attachment

```bash
docker exec enclave_agent ip link show
```

```bash
docker logs xdp-auditor --tail=50
```

You should see startup output similar to:

- `netflow_filter attached to eth0 (mode=drv)`

### B. Verify Prometheus metrics path

```bash
docker exec enclave_agent wget -qO- http://localhost:9400/metrics | head
```

```bash
docker exec enclave_agent wget -qO- http://localhost:9090/-/ready
```

### C. Verify enclave dashboard access (Trust Object)

Open these URLs from your browser:

- `https://grafana.enclave`
- `https://drupal.enclave`

Login to Grafana and validate scrape health + XDP series:

- `xdp_netflow_active_flows`
- `xdp_netflow_packets_total_snapshot`
- `xdp_netflow_flow_packets`

## 5) Teardown

```bash
docker compose down -v
```
