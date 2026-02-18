# Virtualised Zero-DRAM Auditable Trust Stack

This repository now ships a **QEMU SVE2 + Enclave SDN** reference stack that keeps sensitive flow material in vector registers as long as possible before signing.

## Architecture

1. **Virtual hardware (QEMU/SVE2)**
   - `qemu-aarch64-static` emulates an **Arm Neoverse V2** profile (`-cpu neoverse-v2,sve=on,sve2=on`).
   - `mask_sve2.s` performs PII reduction in SVE2 `Z` registers (`z0-z4`) without copying tuples into application heap buffers.
2. **Kernel bridge (XDP -> register path)**
   - XDP parser extracts V7 tuples at line-rate.
   - `bpf_perf_event_output` publishes tuples into perf events consumed by the dispatcher process running in the QEMU guest context.
3. **Evidence signing (PKCS#11)**
   - Dispatcher ingests tuple lanes, applies `mask_v7_tuple_sve2`, hashes evidence, and signs via SoftHSM2 PKCS#11.
   - Signed counters and verification metrics are exported on the enclave-facing Prometheus endpoint.
4. **Isolated services**
   - `qemu-sve` and `xdp-bridge` run on `network_mode: service:enclave`.
   - Drupal/MariaDB remain isolated on `app_net`.

## Delivered files

- `mask_sve2.s` – SVE2 tuple masking routine.
- `docker-compose.yml` – Enclave, XDP bridge, QEMU-SVE dispatcher, SoftHSM2, Prometheus, Drupal, MariaDB.
- `README.md` – WSL2 + runbook instructions.

## WSL2 quick start

### 1) Install prerequisites (including QEMU user emulation)

```bash
sudo apt update
sudo apt install -y \
  qemu-user-static \
  binfmt-support \
  clang llvm gcc make pkg-config \
  libbpf-dev libelf-dev libssl-dev \
  softhsm2 opensc docker.io docker-compose-plugin
```

### 2) Enable and verify `binfmt_misc`

```bash
sudo update-binfmts --enable qemu-aarch64
update-binfmts --display qemu-aarch64
```

### 3) Prepare `.env`

```bash
cat > .env <<'ENV'
ENCLAVE_ENROLMENT_KEY=replace-me
DRUPAL_DB_PASSWORD=drupalpw
MARIADB_ROOT_PASSWORD=rootpw
PKCS11_MODULE_PATH=/usr/lib/softhsm/libsofthsm2.so
PKCS11_PIN=1234
HSM_KEY_LABEL=ato-ed25519
ENV
```

### 4) Initialize SoftHSM2 token

```bash
softhsm2-util --init-token --slot 0 --label ato-token --so-pin 1234 --pin 1234
pkcs11-tool --module /usr/lib/softhsm/libsofthsm2.so \
  --login --pin 1234 \
  --keypairgen --key-type EC:prime256v1 --label ato-ed25519 --usage-sign
```

### 5) Launch stack

```bash
docker compose up -d --build
```

## Register-to-sign path (Zero-DRAM objective)

`NIC/XDP parser -> bpf_perf_event_output -> QEMU SVE2 dispatcher -> Z-register masking -> PKCS#11 signing -> Prometheus over Enclave vNET`

- Tuple fields are emitted from XDP via perf events rather than copied into long-lived userland queues.
- Dispatcher reads perf payloads, maps fields to SVE2 lanes, masks in-register, and immediately signs digests.
- Only signed/hashed artifacts are exported to observability, preserving Trust Object integrity for audit replay.

## OSINT IDE immutable audit flow (CBOR + HSM)

The dispatcher now acts as an **OSINT Sink** for messy forensic leads:

1. Receive raw JSON-like lead blobs (possibly malformed ordering and ad-hoc fields).
2. Apply `mask_v7_sve2` register-first sanitization to IPv4 identifiers and nanosecond timestamps.
3. Call `cbor_wrap_osint_data` to emit **Canonical CBOR** map entries in deterministic key order.
4. Sign the CBOR blob through PKCS#11 using the enclave-secured HSM key.
5. Exfiltrate `(cbor_blob, signature)` records over the enclave SDN path to the centralized Audit Vault file (`AUDIT_VAULT_PATH`).

### Verifying lead authenticity in the IDE

To verify an OSINT lead packet in the IDE:

- Parse the CBOR map and extract the signed byte sequence exactly as persisted.
- Pull the signature companion payload from the Audit Vault frame.
- Verify using the HSM's public key (or issued certificate chain) and the same signature mechanism (`CKM_EDDSA` in this stack).
- Treat only packets with valid signature + trusted key provenance as auditable trust objects.

This ensures analysts can distinguish genuine enclave-originated evidence from tampered or replayed packets.

### Enclave DNS privacy posture

The OSINT IDE should route all resolver traffic through **Enclave DNS** attached to the enclave SDN. This keeps target-domain lookups hidden from the analyst's ISP-facing resolver path:

- DNS requests are emitted inside the enclave network namespace.
- Upstream recursion/DoH egress is pinned to enclave policy endpoints.
- ISP-visible telemetry shows only encrypted tunnel egress, not per-target OSINT domains.

Combined with signed CBOR evidence, this provides both **research confidentiality** and **audit integrity**.

## IPFS-backed Primary Key and enclave-only swarm

The post-signing dispatcher hook now wraps each record as Canonical CBOR IPLD:

```json
{ "data": <CBOR_BLOB>, "signature": <HSM_SIG>, "pubkey": <HSM_KEY_ID> }
```

It then uploads the signed IPLD blob to Kubo (`/api/v0/add?pin=true`) and writes the returned CID into the evidence log as `primary_cid=...`.

- CID becomes the immutable primary key for each trust object.
- Kubo is bound to `${ENCLAVE_VIF_ADDR}` and bootstraps are removed.
- Private swarm mode is enforced via `swarm.key` + `LIBP2P_FORCE_PNET=1` to avoid public DHT leakage.

### Pinning evidence across enclave peers

```bash
ipfs pin add <CID>
```

### Verify evidence signature from IPFS

```bash
ipfs cat <CID> > evidence.ipld.cbor
python3 - <<'PY'
import cbor2
obj = cbor2.load(open("evidence.ipld.cbor", "rb"))
open("payload.cbor", "wb").write(obj["data"])
open("sig.bin", "wb").write(obj["signature"])
PY
openssl dgst -sha256 -verify hsm_pubkey.pem -signature sig.bin payload.cbor
```

## Minimal telemetry folding SNARK example

A runnable micro-batch example is provided in `zk/telemetry-pipeline` with:

- `data/sample_telemetry/batch_1` + `batch_2` containing 32 telemetry packet files,
- witness generation into `data/witnesses/witness_batch_1.json` and `_2.json`,
- Groth16 proving/verification scripts compatible with those witness files,
- optional on-chain commitment using either witness hash or Merkle root.

Run:

```bash
cd zk/telemetry-pipeline
npm install
npm run witness
npm run merkle
npm run snark:compile
npm run snark:setup
node scripts/snark/prove.js data/witnesses/witness_batch_1.json
node scripts/snark/verify.js artifacts/proofs/public_batch_1.json artifacts/proofs/proof_batch_1.json
```
