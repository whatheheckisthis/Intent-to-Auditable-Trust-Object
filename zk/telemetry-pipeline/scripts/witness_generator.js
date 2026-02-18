#!/usr/bin/env node
/**
 * Convert folded telemetry accumulator output into Circom-compatible witness input.
 * Input format: JSON { epochId, lotA: [16 ints], lotB: [16 ints] } or binary file with byte values.
 */
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

function loadAccumulator(inputPath) {
  if (inputPath.endsWith('.json')) {
    return JSON.parse(fs.readFileSync(inputPath, 'utf8'));
  }
  const bytes = fs.readFileSync(inputPath);
  const vals = Array.from(bytes.slice(0, 32));
  return { epochId: 1, lotA: vals.slice(0, 16), lotB: vals.slice(16, 32) };
}

function to512Bits(values) {
  const hash = crypto.createHash('sha512').update(Buffer.from(values)).digest();
  const bits = [];
  for (const b of hash) {
    for (let i = 7; i >= 0; i--) {
      bits.push((b >> i) & 1);
    }
  }
  return bits;
}

function simpleRoot(values) {
  const h = crypto.createHash('sha256').update(Buffer.from(values)).digest('hex');
  return BigInt(`0x${h}`).toString();
}

function main() {
  const root = path.resolve(__dirname, '..');
  const inPath = process.argv[2] || path.join(root, 'data', 'sample_telemetry.json');
  const outPath = process.argv[3] || path.join(root, 'data', 'witness_input.json');

  const telemetry = loadAccumulator(inPath);
  const merged = telemetry.lotA.concat(telemetry.lotB);
  if (merged.length !== 32) {
    throw new Error(`Expected 32 telemetry entries (2x16), got ${merged.length}`);
  }

  const witnessBits = to512Bits(merged);
  const midpoint = 256;
  const allowedDivergence = 128;

  const rootA = simpleRoot(telemetry.lotA);
  const rootB = simpleRoot(telemetry.lotB);

  // This matches the Poseidon fold ordering in telemetry_folding.circom.
  // We leave claimedCombinedRoot as placeholder for demo and update it during proving setup.
  const input = {
    witnessBits,
    midpoint,
    allowedDivergence,
    microBatchRoots: [rootA, rootB],
    claimedCombinedRoot: '0'
  };

  fs.writeFileSync(outPath, JSON.stringify(input, null, 2));
  console.log(`Wrote witness input to ${outPath}`);
}

main();
