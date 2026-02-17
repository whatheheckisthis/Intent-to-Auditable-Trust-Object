# Auditable Trust Object (ATO) for Drupal + MariaDB

This stack delivers a local, high-throughput monitoring plane for Drupal/MariaDB with a trust chain:

**NIC (hardware)** → **XDP kernel hook (L1)** → **Per-CPU eBPF maps (L2)** → **SIMD masking** → **PKCS#11/HSM signatures (L3)** → **Enclave tunnel + Prometheus metrics**.

## Dependency discovery and configuration

Install runtime/build dependencies (Ubuntu/WSL2):

```bash
sudo apt update
sudo apt install -y clang llvm libbpf-dev linux-headers-$(uname -r) \
  gcc make nasm pkg-config libssl-dev softhsm2 opensc docker.io docker-compose-plugin
```

Initialize a SoftHSM2 token and Ed25519 key:

```bash
softhsm2-util --init-token --slot 0 --label ato-token --so-pin 1234 --pin 1234
pkcs11-tool --module /usr/lib/softhsm/libsofthsm2.so --login --pin 1234 \
  --keypairgen --key-type EC:edwards25519 --label ato-ed25519 --usage-sign
```

Create env file:

```bash
cp .env.example .env
```

Add/update these values in `.env`:

```dotenv
ENCLAVE_ENROLMENT_KEY=...
DRUPAL_DB_PASSWORD=...
MARIADB_ROOT_PASSWORD=...
PKCS11_MODULE_PATH=/usr/lib/softhsm/libsofthsm2.so
PKCS11_PIN=1234
HSM_KEY_LABEL=ato-ed25519
```

## Core files

- `xdp_audit.c`: XDP NetFlow parser for IPv4/IPv6, per-CPU hash flow accounting, ring-buffer audit events.
- `fast_mask.asm`: SIMD masking (`movdqu` + `pand`) for address privacy.
- `dispatcher.c`: Per-CPU map aggregation, mandatory masking call per flow, PKCS#11 signing, Prometheus exposition.
- `docker-compose.yml`: Enclave fabric + privileged dispatcher + local Drupal stack on `app_net`.

## Bring up the stack

```bash
docker compose up -d --build
```

## Verify trust chain

1) **XDP loaded**
```bash
docker logs xdp-auditor --tail=100
```

2) **Dispatcher producing signed evidence**
```bash
docker logs ato-dispatcher --tail=100
```

3) **Prometheus metrics over enclave interface**
```bash
docker exec enclave_agent wget -qO- http://localhost:9400/metrics
```

## Verify immutable audit log signatures

Extract one evidence line's digestable payload and verify with OpenSSL (example flow):

```bash
openssl dgst -sha256 -verify public_key.pem -signature flow.sig flow_payload.bin
```

Expected output:

```text
Verified OK
```

## WSL2 quick start

1. Use WSL2 kernel with eBPF support (`uname -r`).
2. Start Docker Desktop with WSL integration.
3. Install dependencies listed above inside WSL distro.
4. Build and run `docker compose up -d --build`.
5. Generate traffic to Drupal and inspect `/var/log/audit/signed_evidence.log` in dispatcher container.
