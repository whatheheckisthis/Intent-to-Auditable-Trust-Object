#!/usr/bin/env node
/**
 * Generate a Groth16 proof from a Circom witness input JSON.
 * Usage:
 *   node scripts/prove.js data/witnesses/witness_batch_1.json
 */
const { execSync } = require('child_process');
const path = require('path');
const fs = require('fs');

const root = path.resolve(__dirname, '..');
const buildDir = path.join(root, 'artifacts', 'build', 'telemetry_folding_js');
const zkeyFinal = path.join(root, 'artifacts', 'zkey', 'telemetry_final.zkey');
const outDir = path.join(root, 'artifacts', 'proofs');
fs.mkdirSync(outDir, { recursive: true });

const inputJson = process.argv[2] || path.join(root, 'data', 'witnesses', 'witness_batch_1.json');
const tag = path.basename(inputJson, '.json').replace(/^witness_/, '');
const witnessWtns = path.join(outDir, `witness_${tag}.wtns`);
const proofJson = path.join(outDir, `proof_${tag}.json`);
const publicJson = path.join(outDir, `public_${tag}.json`);

execSync(
  `node ${path.join(buildDir, 'generate_witness.js')} ${path.join(buildDir, 'telemetry_folding.wasm')} ${inputJson} ${witnessWtns}`,
  { stdio: 'inherit' }
);
execSync(`snarkjs groth16 prove ${zkeyFinal} ${witnessWtns} ${proofJson} ${publicJson}`, { stdio: 'inherit' });

// Backward-compatible aliases for tooling that expects non-suffixed names.
fs.copyFileSync(proofJson, path.join(outDir, 'proof.json'));
fs.copyFileSync(publicJson, path.join(outDir, 'public.json'));

console.log(`Proof generated:\n- ${proofJson}\n- ${publicJson}`);
