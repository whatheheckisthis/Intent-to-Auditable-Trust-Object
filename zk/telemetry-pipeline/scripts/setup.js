#!/usr/bin/env node
/**
 * Groth16 setup for telemetry folding circuit.
 * Produces ptau artifacts, final zkey, verifier key, and Solidity verifier contract.
 */
const { execSync } = require('child_process');
const path = require('path');
const fs = require('fs');

const root = path.resolve(__dirname, '..');
const buildDir = path.join(root, 'artifacts', 'build');
const zkeyDir = path.join(root, 'artifacts', 'zkey');
fs.mkdirSync(zkeyDir, { recursive: true });

const r1cs = path.join(buildDir, 'telemetry_folding.r1cs');
const ptau0 = path.join(zkeyDir, 'powersOfTau28_hez_final_12.ptau');
const zkey0 = path.join(zkeyDir, 'telemetry_0000.zkey');
const zkeyFinal = path.join(zkeyDir, 'telemetry_final.zkey');
const vkey = path.join(zkeyDir, 'verification_key.json');
const verifierSol = path.join(root, 'contracts', 'Groth16Verifier.sol');

execSync(`snarkjs groth16 setup ${r1cs} ${ptau0} ${zkey0}`, { stdio: 'inherit' });
execSync(`snarkjs zkey contribute ${zkey0} ${zkeyFinal} --name="Telemetry ceremony" -v -e="telemetry entropy"`, { stdio: 'inherit' });
execSync(`snarkjs zkey export verificationkey ${zkeyFinal} ${vkey}`, { stdio: 'inherit' });
execSync(`snarkjs zkey export solidityverifier ${zkeyFinal} ${verifierSol}`, { stdio: 'inherit' });

console.log('Groth16 setup complete.');
