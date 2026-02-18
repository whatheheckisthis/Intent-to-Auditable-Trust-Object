# Telemetry Folding SNARK Pipeline

This folder provides a reproducible end-to-end flow for a hardware-accelerated telemetry folding pipeline:

1. ingest two micro-batches of telemetry (2 x 16 = 32 entries),
2. fold into a 512-bit witness,
3. prove constraints in Circom/Groth16,
4. verify locally with snarkjs,
5. submit verification result to an Ethereum syslog-style audit contract.

## Folder layout

- `circuits/telemetry_folding.circom`: Main circuit with modular micro-batch root folding.
- `scripts/compile.js`: Compiles Circom into R1CS/WASM artifacts.
- `scripts/setup.js`: Produces zkey + verification key + Solidity verifier.
- `scripts/witness_generator.js`: Converts hardware output into Circom input JSON.
- `scripts/prove.js`: Generates witness/proof/public signals.
- `scripts/verify.js`: Verifies proof against exported verification key.
- `scripts/merkle_helper.js`: Computes Merkle roots for micro-batches.
- `contracts/TelemetryProof.sol`: Ethereum audit contract emitting syslog-like events.
- `scripts/deploy_contract.js`: Deploys the audit contract with ethers.js.
- `scripts/submit_proof.js`: Submits proof + witness hash.
- `scripts/verify_on_chain.js`: Reads verification status for a sequence number.
- `notebooks/*.ipynb`: Four notebooks that document witness, proving, and on-chain verification.

## Dependencies

- Node.js 18+
- `circom` compiler installed in PATH
- `snarkjs` and `ethers` (`npm install` in this directory)
- Solidity toolchain (Hardhat/Foundry/solc) for compiling and exporting `artifacts/TelemetryProof.json`

## Quickstart

```bash
cd zk/telemetry-pipeline
npm install
node scripts/witness_generator.js data/sample_telemetry.json data/witness_input.json
node scripts/compile.js
node scripts/setup.js
node scripts/prove.js data/witness_input.json
node scripts/verify.js
```

## Public input conventions

- Circom output `valid` is included in `public.json`.
- `combinedRoot` and `foldedWeight` are also public outputs to support auditor introspection.
- `submit_proof.js` sends `public.json` as `pubSignals` to `TelemetryProof.submitProof`.

## Ethereum flow

1. Compile `contracts/TelemetryProof.sol` and generated `contracts/Groth16Verifier.sol`.
2. Produce `artifacts/TelemetryProof.json` (ABI + bytecode).
3. Deploy verifier + TelemetryProof with `scripts/deploy_contract.js`.
4. Submit proof via `scripts/submit_proof.js`.
5. Auditors call `results(seqNo)` or inspect `SyslogAudit` events.

## Notes on 512-bit succinctness

The witness representation is fixed at 512 bits (`witnessBits[512]`) and proving uses Groth16 for succinct proofs. You can extend `TelemetryFolding(batchCount)` by changing `batchCount` and regenerating artifacts.
