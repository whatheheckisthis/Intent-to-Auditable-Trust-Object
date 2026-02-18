#!/usr/bin/env node
/**
 * Verify a Groth16 proof against the exported verification key.
 * Usage:
 *   node scripts/verify.js [public_json] [proof_json]
 */
const { execSync } = require('child_process');
const path = require('path');

const root = path.resolve(__dirname, '..');
const vkey = path.join(root, 'artifacts', 'zkey', 'verification_key.json');
const publicJson = process.argv[2] || path.join(root, 'artifacts', 'proofs', 'public.json');
const proofJson = process.argv[3] || path.join(root, 'artifacts', 'proofs', 'proof.json');

execSync(`snarkjs groth16 verify ${vkey} ${publicJson} ${proofJson}`, { stdio: 'inherit' });
