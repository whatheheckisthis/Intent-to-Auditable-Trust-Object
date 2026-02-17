# Auditable Trust Object (ATO)

This repository implements a full-stack **Auditable Trust Object** that binds together hardware parsing, eBPF enforcement, SIMD privacy controls, and HSM-backed signatures.

## Source layers delivered

- **Hardware (`hardware/netflow_parser.v`)**
  - 512-bit AXI-stream parser for 100Gbps pipelines.
  - Extracts V7 tuple fields (src/dst IP, src/dst port, protocol, ToS, TCP flags).
  - Carries NIC hardware timestamp metadata for downstream attestation.
- **Kernel (`xdp_audit.c`)**
  - XDP L1 filter parses IPv4/IPv6 TCP/UDP flows.
  - L2 accounting map uses `BPF_MAP_TYPE_PERCPU_HASH` and emits ring-buffer alerts.
- **Evidence (`dispatcher.c`, `fast_mask.asm`, `pkcs11_signer.c`)**
  - SIMD assembly masking (`/24` IPv4, `/64` IPv6) before hashing.
  - Batch signing path (`1000 flows/call`) to amortize HSM round-trip overhead.
  - PKCS#11 signing via SoftHSM2 or physical HSM.
- **Infrastructure (`docker-compose.yml`)**
  - Enclave secure fabric and privileged dispatcher with `/sys/fs/bpf` mounted.
  - SoftHSM2 + observability (Prometheus/Grafana).
  - Isolated local stack (`Drupal + MariaDB + Apache + Nginx`) on `app_net`.

## Full trust chain

Evidence is generated and linked in the following chain:

`[NIC Hardware Timestamp] -> [XDP Flow Hash Context] -> [BPF Map Checksum] -> [HSM Signature]`

1. Hardware parser emits tuple + NIC timestamp metadata.
2. XDP parser records per-flow counters in pinned per-CPU maps.
3. Dispatcher folds map contents into `bpf_map_checksum` and builds evidence digest.
4. Batch PKCS#11 signer produces token-backed signatures over SHA-256 evidence digests.

## WSL2 deployment

### 1) Install required packages

```bash
sudo apt update
sudo apt install -y clang llvm gcc make nasm pkg-config \
  libbpf-dev libelf-dev libssl-dev softhsm2 opensc \
  docker.io docker-compose-plugin linux-headers-$(uname -r)
```

### 2) Configure environment

```bash
cat > .env <<'ENV'
ENCLAVE_ENROLMENT_KEY=replace-me
DRUPAL_DB_PASSWORD=drupalpw
MARIADB_ROOT_PASSWORD=rootpw
PKCS11_MODULE_PATH=/usr/lib/softhsm/libsofthsm2.so
PKCS11_PIN=1234
HSM_KEY_LABEL=ato-ed25519
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=admin
ENV
```

### 3) Initialize SoftHSM2 token

```bash
softhsm2-util --init-token --slot 0 --label ato-token --so-pin 1234 --pin 1234
pkcs11-tool --module /usr/lib/softhsm/libsofthsm2.so \
  --login --pin 1234 \
  --keypairgen --key-type EC:edwards25519 --label ato-ed25519 --usage-sign
```

### 4) Build and run

```bash
docker compose up -d --build
```

### 5) Optional local binary build

```bash
make dispatcher LDFLAGS="-lbpf"
```

## Runtime verification

Signed records are appended to `/var/log/audit/signed_evidence.log` and include:

- `bpf_map_checksum`
- `digest_sha256`
- `verified`
- `signature` (base64)

To verify digest/signature pairs, export the HSM public key and use `openssl pkeyutl -verify` on the digest bytes.
