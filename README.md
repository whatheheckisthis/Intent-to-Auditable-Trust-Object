# Auditable Trust Object (ATO) Stack

This repository now includes an end-to-end **Auditable Trust Object** pipeline for high-throughput flow capture and signed evidence export:

1. **Kernel capture (L1/L2):** XDP NetFlow parser + Per-CPU hash map accounting.
2. **Privacy + signing (L4/L5):** SIMD masking + PKCS#11 HSM signatures.
3. **Secure transport:** Dispatcher, Prometheus, and Grafana on the Enclave SDN namespace.
4. **App stack:** Drupal + MariaDB + Nginx isolated on `app_net`.

## Key source files

- `xdp_audit.c` and `ebpf/xdp_audit.c`:
  - Parses IPv4/IPv6 TCP+UDP packets at XDP hook.
  - Maintains per-flow counters in `BPF_MAP_TYPE_PERCPU_HASH` (`flow_map`).
  - Emits burst audit events via ring buffer.
- `dispatcher.c`:
  - Reads and aggregates per-CPU map values.
  - Builds `evidence_t` records.
  - Calls SIMD masking before hashing/signing.
  - Exposes Prometheus metrics including verification success ratio.
- `fast_mask.asm`:
  - SSE (`movdqu`/`pand`) masking routine for IPv4 (/24) and IPv6 (/64).
- `pkcs11_signer.c` + `pkcs11_signer.h`:
  - PKCS#11 session management against real HSM or SoftHSM2.
  - Signs SHA-256 digest bytes using token private key.
- `docker-compose.yml`:
  - Enclave, XDP auditor, dispatcher, SoftHSM2, Prometheus, Grafana.
  - Drupal, MariaDB, Apache, Nginx on isolated `app_net`.
- `grafana/dashboards/ato-signature-verification.json`:
  - **Auditable Flow Rate** panel.
  - **Signature Verification Success** panel.

## WSL2 quick start

1. **Install WSL2 + Ubuntu** and verify kernel:
   ```bash
   uname -r
   ```
2. **Install build/runtime dependencies inside WSL2:**
   ```bash
   sudo apt update
   sudo apt install -y clang llvm gcc make nasm pkg-config \
     libbpf-dev libelf-dev libssl-dev softhsm2 opensc \
     docker.io docker-compose-plugin linux-headers-$(uname -r)
   ```
3. **Enable Docker Desktop WSL integration** (or run docker engine in WSL).
4. **Create `.env`**:
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
5. **Initialize SoftHSM2 token + signing key:**
   ```bash
   softhsm2-util --init-token --slot 0 --label ato-token --so-pin 1234 --pin 1234
   pkcs11-tool --module /usr/lib/softhsm/libsofthsm2.so \
     --login --pin 1234 \
     --keypairgen --key-type EC:edwards25519 --label ato-ed25519 --usage-sign
   ```
6. **Start stack:**
   ```bash
   docker compose up -d --build
   ```

## Observability endpoints

- Dispatcher metrics: `http://localhost:9400/metrics` (inside enclave network namespace).
- Prometheus: `http://localhost:9090` (inside enclave namespace).
- Grafana: `http://localhost:3000` (inside enclave namespace), dashboard title: **Auditable Trust Object Overview**.

## Verify signed evidence with OpenSSL

Dispatcher writes lines like:

```text
ts=... digest_sha256=<hex> verified=1 signature=<base64>
```

To verify a record:

1. Extract `digest_sha256` (hex) and `signature` (base64) from one log line.
2. Convert digest hex to binary:
   ```bash
   echo -n "<digest_sha256_hex>" | xxd -r -p > digest.bin
   ```
3. Convert signature base64 to raw bytes:
   ```bash
   echo -n "<signature_base64>" | base64 -d > flow.sig
   ```
4. Export public key from token (example with pkcs11-tool + OpenSSL workflow).
5. Verify:
   ```bash
   openssl pkeyutl -verify -pubin -inkey public_key.pem \
     -rawin -in digest.bin -sigfile flow.sig
   ```

`Signature Verified Successfully` indicates the evidence record matches the signed digest.
