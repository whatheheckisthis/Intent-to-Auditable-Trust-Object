# Auditable Trust Object: Enclave + eBPF/XDP Monitoring Stack

This repository ships a secure and auditable monitoring architecture that combines:

- **Line-rate kernel telemetry** from an XDP/eBPF audit hook (`xdp_audit.c`),
- **An enclave-secured userspace sidecar** that exports Prometheus metrics,
- **A private overlay network backbone** for observability and dashboard access,
- **An isolated app plane** (`Nginx -> Drupal -> MariaDB`) on `app_net`.

## 1) Architecture

### Software-Verilog Layer (Kernel/NIC path)

- `ebpf/xdp_audit.c` attaches at the XDP hook and is intended for **`XDP_DRV`** (driver mode) and optionally **`XDP_HW`** where NIC offload is supported.
- The program performs a packet-timing audit and emits events through a `BPF_MAP_TYPE_PERF_EVENT_ARRAY` map named `audit_events`.
- The TVLA-inspired logic computes an online timing baseline (mean + variance via Welford updates) and flags packets where the absolute timing deviation yields a high t-score (`|t| >= 4`) after minimum samples are collected.

### Bridge Layer (Userspace sidecar)

- `ebpf/agent.py` dynamically loads and attaches the XDP program.
- The sidecar consumes perf events from the eBPF map and exposes `/metrics` on port `9400`.
- Container runs with:
  - `network_mode: "service:enclave"`
  - `privileged: true`
  - host mounts for `/sys/kernel/debug`, `/lib/modules`, `/usr/src`.

### Hybrid Compose Topology

- **Enclave service** provides the secure networking backbone and `.enclave` name resolution.
- **Prometheus + Grafana** run in the enclave network namespace and scrape the XDP sidecar privately.
- **App stack** (`drupal`, `apache`, `db`) remains isolated on `app_net`.
- **Gateway Nginx** is attached to the enclave namespace and exposes TLS on host port `443`.

## 2) Key Files

- `ebpf/xdp_audit.c`
- `ebpf/agent.py`
- `ebpf/Dockerfile`
- `docker-compose.yml`
- `prometheus.yml`

## 3) WSL2 Quick Start

1. **Prerequisites**

   - Docker Desktop with WSL2 backend enabled.
   - Linux kernel with eBPF/XDP support in your WSL distro.
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

- `XDP audit hook attached on eth0 with mode=drv`

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

- `xdp_audit_events_total`
- `xdp_audit_leak_flags_total`
- `xdp_audit_last_tscore`

## 5) Teardown

```bash
docker compose down -v
```
