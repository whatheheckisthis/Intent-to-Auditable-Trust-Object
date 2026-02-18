#!/usr/bin/env node
/**
 * Generate a Groth16 proof from a Circom witness input JSON.
 * Usage: node scripts/prove.js data/witness_input.json
 */
const { execSync } = require('child_process');
const path = require('path');
const fs = require('fs');

const root = path.resolve(__dirname, '..');
const buildDir = path.join(root, 'artifacts', 'build', 'telemetry_folding_js');
const zkeyFinal = path.join(root, 'artifacts', 'zkey', 'telemetry_final.zkey');
const outDir = path.join(root, 'artifacts', 'proofs');
fs.mkdirSync(outDir, { recursive: true });

const inputJson = process.argv[2] || path.join(root, 'data', 'witness_input.json');
const witnessWtns = path.join(outDir, 'witness.wtns');
const proofJson = path.join(outDir, 'proof.json');
const publicJson = path.join(outDir, 'public.json');

execSync(
  `node ${path.join(buildDir, 'generate_witness.js')} ${path.join(buildDir, 'telemetry_folding.wasm')} ${inputJson} ${witnessWtns}`,
  { stdio: 'inherit' }
);
execSync(`snarkjs groth16 prove ${zkeyFinal} ${witnessWtns} ${proofJson} ${publicJson}`, { stdio: 'inherit' });

console.log(`Proof generated:\n- ${proofJson}\n- ${publicJson}`);
