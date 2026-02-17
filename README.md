# Auditable Trust Object (Drupal/MariaDB)

High-performance monitoring stack for a local Drupal + MariaDB deployment, with cryptographic evidence signing.

## Trust Chain

1. **Hardware NIC** receives packets at line rate.
2. **XDP (`ebpf/xdp_audit.c`)** performs L1 filtering and L2 per-CPU flow accounting (`flow_v4_map`, `flow_v6_map`).
3. **SIMD Masking (`fast_mask.asm`)** masks source/destination IP pairs via `movdqu` + `pand` for privacy-by-design.
4. **Dispatcher (`dispatcher.c`)** aggregates map snapshots, emits Prometheus metrics, hashes evidence (SHA-256), and signs using PKCS#11 (SoftHSM2-compatible).
5. **Enclave Tunnel (`docker-compose.yml`)** publishes telemetry through the enclave namespace while the app plane stays isolated on `app_net`.

## Files Delivered

- `ebpf/xdp_audit.c`
- `fast_mask.asm`
- `dispatcher.c`
- `docker-compose.yml`
- `README.md`

## WSL2 Quick Start

```bash
cp .env.example .env
```

Set required values in `.env`:

- `ENCLAVE_ENROLMENT_KEY`
- `PKCS11_PIN`
- `DRUPAL_DB_PASSWORD`
- `MARIADB_ROOT_PASSWORD`

Build and run:

```bash
docker compose up -d --build
```

## Validate Monitoring + Signing

Check dispatcher metrics from enclave namespace:

```bash
docker exec enclave wget -qO- http://127.0.0.1:9108/metrics
```

Tail immutable log:

```bash
docker exec auditable-dispatcher tail -f /workspace/logs/immutable_audit.log
```

## Verify Immutable Audit Log Signature

1. Extract one evidence line and Base64 signature.
2. Decode signature and verify against the dispatcher digest.

Example workflow:

```bash
jq -r '.evidence' logs/immutable_audit.log | head -n1 > /tmp/evidence.json
jq -r '.signature_b64' logs/immutable_audit.log | head -n1 | base64 -d > /tmp/evidence.sig
openssl dgst -sha256 -verify /path/to/public.pem -signature /tmp/evidence.sig /tmp/evidence.json
```

Expected result:

- `Verified OK`
