#!/usr/bin/env node
/**
 * Verify a Groth16 proof against the exported verification key.
 * Usage: node scripts/verify.js
 */
const { execSync } = require('child_process');
const path = require('path');

const root = path.resolve(__dirname, '..');
const vkey = path.join(root, 'artifacts', 'zkey', 'verification_key.json');
const publicJson = path.join(root, 'artifacts', 'proofs', 'public.json');
const proofJson = path.join(root, 'artifacts', 'proofs', 'proof.json');

execSync(`snarkjs groth16 verify ${vkey} ${publicJson} ${proofJson}`, { stdio: 'inherit' });
