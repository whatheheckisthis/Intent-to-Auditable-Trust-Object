# Auditable Trust Object (ATO) Stack

This stack combines three layers:

1. **Layer 1 (Kernel):** `ebpf/xdp_audit.c` captures line-rate NetFlow tuples at XDP hook speed.
2. **Layer 2 (Per-CPU State):** `flow_map` uses `BPF_MAP_TYPE_PERCPU_HASH` to keep lockless per-CPU counters.
3. **Layer 3 (Dispatcher):** `dispatcher.c` aggregates per-CPU values, masks sensitive fields, signs evidence via PKCS#11/HSM, and exports Prometheus counters.

## Layer 1 and Layer 2 details

`ebpf/xdp_audit.c` implements:

- `SEC("xdp") int netflow_filter(...)` to parse Ethernet + IPv4 + TCP/UDP tuples.
- `flow_map` as a pinned per-CPU hash map (`LIBBPF_PIN_BY_NAME`) for shared userspace reads from `/sys/fs/bpf/flow_map`.
- packet/byte accounting and spike guards using `spike_guard` + ring buffer events for anomaly hints.

## Layer 3 userspace dispatcher

`dispatcher.c` loop:

1. Opens pinned map at `FLOW_MAP_PATH` (default `/sys/fs/bpf/flow_map`).
2. Iterates all keys with `bpf_map_get_next_key`.
3. Reads per-CPU values and sums packet/byte totals.
4. Calls `mask_sensitive_fields` to remove PII-like fields (IP precision and L4 ports).
5. Hashes evidence (`SHA256`) and performs HSM signing retries through `sign_evidence`.
6. Appends signed records to `/var/log/audit/signed_evidence.log`.
7. Exposes Prometheus counters on `:9400`:
   - `ato_verified_flows_total`
   - `ato_unverified_flows_total`
   - `ato_signed_evidence_total`

## Docker architecture

`docker-compose.yml` services:

- **enclave**: secure fabric with `NET_ADMIN` capability.
- **xdp-auditor**: privileged eBPF loader + collector sharing the enclave network namespace.
- **dispatcher**: privileged layer-3 signer, mounts `/sys/fs/bpf` and writes signed audit evidence.
- **prometheus/grafana**: scrape + visualize metrics through enclave network namespace.
- **local app stack on isolated `app_net`**: `nginx`, `apache`, `drupal`, `db` (MariaDB).

## Persistence & audit log immutability

- `signed-audit-log` volume is mounted at `/var/log/audit`.
- Dispatcher writes append-only evidence at:
  - `/var/log/audit/signed_evidence.log`

Recommended hardening in the dispatcher container (or host bind mount):

```bash
chattr +a /var/log/audit/signed_evidence.log
```

This enforces append-only semantics for tamper resistance.

## Grafana dashboard

Dashboard JSON:

- `grafana/dashboards/ato-signature-verification.json`

Main panel: **Verified vs Unverified Flows** pie chart from:

- `ato_verified_flows_total`
- `ato_unverified_flows_total`

## WSL2 quick start

Inside your WSL distro (Ubuntu example):

```bash
sudo apt-get update
sudo apt-get install -y build-essential clang llvm libelf-dev libssl-dev libbpf-dev libpkcs11-helper1-dev pkg-config
```

Build dispatcher locally:

```bash
make clean
make dispatcher LDFLAGS="-lbpf"
```

### Open signed evidence log

```bash
sudo tail -f /var/log/audit/signed_evidence.log
```

### Verify a signature using OpenSSL

Extract base64 signature from a log line into `sig.b64` and evidence payload bytes into `evidence.bin`:

```bash
base64 -d sig.b64 > sig.bin
openssl dgst -sha256 -verify public.pem -signature sig.bin evidence.bin
```

Expected output when valid: `Verified OK`.
