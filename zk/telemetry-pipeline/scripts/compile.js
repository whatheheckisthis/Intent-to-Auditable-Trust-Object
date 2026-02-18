#!/usr/bin/env node
/**
 * Compile the telemetry folding Circom circuit into R1CS/WASM/Sym artifacts.
 */
const { execSync } = require('child_process');
const path = require('path');
const fs = require('fs');

const root = path.resolve(__dirname, '..');
const circuit = path.join(root, 'circuits', 'telemetry_folding.circom');
const outDir = path.join(root, 'artifacts', 'build');

fs.mkdirSync(outDir, { recursive: true });

const cmd = [
  'circom',
  circuit,
  '--r1cs',
  '--wasm',
  '--sym',
  '-o',
  outDir,
].join(' ');

console.log(`Compiling circuit: ${cmd}`);
execSync(cmd, { stdio: 'inherit' });
console.log('Circuit compilation complete.');
