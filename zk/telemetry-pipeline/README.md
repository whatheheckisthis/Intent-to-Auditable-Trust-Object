# Telemetry Folding SNARK Pipeline

This folder provides a reproducible end-to-end flow for a hardware-accelerated telemetry folding pipeline:

1. ingest 32 telemetry packets split into two micro-batches,
2. fold each batch into a normalized (mod 3329) 16-coefficient witness vector,
3. generate Circom input JSON, prove constraints in Circom/Groth16,
4. verify locally with snarkjs,
5. optionally commit witness hash or Merkle root on Ethereum.

## Folder layout

- `circuits/telemetry_folding.circom`: Main circuit with modular micro-batch root folding.
- `data/sample_telemetry/batch_1|batch_2/item_*.json`: Minimal 32-item telemetry dataset.
- `data/witnesses/witness_batch_1.json`, `data/witnesses/witness_batch_2.json`: Generated batch witnesses.
- `scripts/witness_generator.js`: Folds telemetry packets into Circom-compatible witness JSON.
- `scripts/merkle_helper.js`: Computes keccak256 Merkle roots per witness batch.
- `scripts/snark/*.js`: SNARK workflow wrappers (compile/setup/prove/verify).
- `scripts/eth/*.js`: Ethereum submission / status wrappers.
- `scripts/prove.js`: Generates witness/proof/public signals from a witness JSON.
- `scripts/verify.js`: Verifies proof against exported verification key.
- `contracts/TelemetryProof.sol`: Ethereum audit contract emitting syslog-like events.

## Dataset format

Each telemetry packet file stores one vector:

```json
{
  "coefficients": [123, 45, 678, 234, 12, 89, 345, 2, 90, 456, 78, 12, 34, 567, 89, 23]
}
```

Constraints used by the generator:

- coefficient count = `16`
- values normalized in `Z_3329`
- micro-batch fold = element-wise sum mod 3329 over each batch

## Quickstart (minimal 32-file example)

```bash
cd zk/telemetry-pipeline
npm install
npm run witness                # writes data/witnesses/witness_batch_1.json and _2.json
npm run merkle                 # prints/writes batch Merkle roots
npm run snark:compile
npm run snark:setup
node scripts/snark/prove.js data/witnesses/witness_batch_1.json
node scripts/snark/verify.js artifacts/proofs/public_batch_1.json artifacts/proofs/proof_batch_1.json
```

## Telemetry → witness → proof → chain logging

1. **Telemetry fold**: `scripts/witness_generator.js` reads `data/sample_telemetry/batch_*` and builds `witness_batch_X.json`.
2. **Commitment**: `scripts/merkle_helper.js` computes keccak256 batch roots for optional audit logging.
3. **Proof**: `scripts/prove.js` uses Circom witness generation + `snarkjs groth16 prove`.
4. **Local verification**: `scripts/verify.js` runs `snarkjs groth16 verify`.
5. **On-chain logging**:
   - `scripts/eth/submit_proof.js` supports `WITNESS_HASH` or `MERKLE_ROOT` environment overrides.
   - `scripts/eth/verify_on_chain.js` reads verification status and stored commitment hash.

## Public input conventions

- Circom output `valid` is included in `public*.json`.
- `combinedRoot` and `foldedWeight` are public outputs to support auditor introspection.
- `submit_proof.js` sends public signals to `TelemetryProof.submitProof`.

## Notes on succinctness

The witness bit representation is fixed at 512 bits (`witnessBits[512]`) while adding a 16-element folded witness vector for micro-batch compatibility. You can extend `TelemetryFolding(batchCount, witnessWidth)` by changing parameters and regenerating artifacts.
