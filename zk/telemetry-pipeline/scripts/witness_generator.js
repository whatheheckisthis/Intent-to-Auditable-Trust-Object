#!/usr/bin/env node
/**
 * Build per-batch Circom-compatible witness input from telemetry packet JSON files.
 *
 * Supported invocation patterns:
 *  1) node scripts/witness_generator.js
 *     - Reads data/sample_telemetry/{batch_1,batch_2}/item_*.json
 *     - Writes data/witnesses/witness_batch_{1,2}.json
 *
 *  2) node scripts/witness_generator.js <input_json> <output_json>
 *     - Backwards-compatible single-file mode for sample_telemetry.json
 */
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const MODULUS = 3329;
const MIDPOINT = 256;
const ALLOWED_DIVERGENCE = 128;
const EXPECTED_COEFFICIENTS = 16;

function modNormalize(value, modulus = MODULUS) {
  const n = Number(value);
  if (!Number.isInteger(n)) {
    throw new Error(`Telemetry coefficient must be an integer, got ${value}`);
  }
  return ((n % modulus) + modulus) % modulus;
}

function hashToBits(values) {
  const bytes = values.map((v) => modNormalize(v));
  const hash = crypto.createHash('sha512').update(Buffer.from(bytes)).digest();
  const bits = [];
  for (const b of hash) {
    for (let i = 7; i >= 0; i--) {
      bits.push((b >> i) & 1);
    }
  }
  return bits;
}

function buildWitnessFromCoefficients(coefficients) {
  const normalized = coefficients.map((v) => modNormalize(v));
  if (normalized.length !== EXPECTED_COEFFICIENTS) {
    throw new Error(`Expected ${EXPECTED_COEFFICIENTS} coefficients, got ${normalized.length}`);
  }

  return {
    witness: normalized,
    witnessBits: hashToBits(normalized),
    midpoint: MIDPOINT,
    allowedDivergence: ALLOWED_DIVERGENCE,
    microBatchRoots: [0],
    claimedCombinedRoot: 0,
  };
}

function loadLegacySingleFile(inputPath) {
  const telemetry = JSON.parse(fs.readFileSync(inputPath, 'utf8'));
  const merged = [...(telemetry.lotA || []), ...(telemetry.lotB || [])].map((v) => modNormalize(v));
  if (merged.length !== 32) {
    throw new Error(`Expected 32 telemetry entries (2x16), got ${merged.length}`);
  }

  // Keep legacy behavior: hash 32 bytes into 512 witness bits.
  return {
    witnessBits: hashToBits(merged),
    midpoint: MIDPOINT,
    allowedDivergence: ALLOWED_DIVERGENCE,
    microBatchRoots: [0, 0],
    claimedCombinedRoot: 0,
  };
}

function sortedTelemetryItems(batchDir) {
  return fs.readdirSync(batchDir)
    .filter((name) => /^item_\d+\.json$/.test(name))
    .sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
}

function generateFromDirectoryTree(baseTelemetryDir, outDir) {
  fs.mkdirSync(outDir, { recursive: true });

  const batchDirs = fs.readdirSync(baseTelemetryDir)
    .filter((name) => name.startsWith('batch_'))
    .sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));

  if (batchDirs.length === 0) {
    throw new Error(`No batch directories found under ${baseTelemetryDir}`);
  }

  for (const batchName of batchDirs) {
    const batchDir = path.join(baseTelemetryDir, batchName);
    const stat = fs.statSync(batchDir);
    if (!stat.isDirectory()) continue;

    const items = sortedTelemetryItems(batchDir);
    if (items.length === 0) {
      throw new Error(`No telemetry items found in ${batchDir}`);
    }

    items.forEach((itemFile) => {
      const packet = JSON.parse(fs.readFileSync(path.join(batchDir, itemFile), 'utf8'));
      if (!Array.isArray(packet.coefficients)) {
        throw new Error(`${itemFile} is missing a coefficients array`);
      }
    });

    const folded = new Array(EXPECTED_COEFFICIENTS).fill(0);
    for (const itemFile of items) {
      const packet = JSON.parse(fs.readFileSync(path.join(batchDir, itemFile), 'utf8'));
      const coefficients = packet.coefficients.map((v) => modNormalize(v));
      if (coefficients.length !== EXPECTED_COEFFICIENTS) {
        throw new Error(`${itemFile} expected ${EXPECTED_COEFFICIENTS} coefficients, got ${coefficients.length}`);
      }
      for (let i = 0; i < EXPECTED_COEFFICIENTS; i++) {
        folded[i] = modNormalize(folded[i] + coefficients[i]);
      }
    }

    const batchIndex = Number(batchName.split('_').pop());
    const witness = buildWitnessFromCoefficients(folded);
    const outPath = path.join(outDir, `witness_batch_${batchIndex}.json`);
    fs.writeFileSync(outPath, JSON.stringify(witness, null, 2));
    console.log(`Wrote ${outPath} (${items.length} telemetry items folded)`);
  }
}

function main() {
  const root = path.resolve(__dirname, '..');
  const arg1 = process.argv[2];
  const arg2 = process.argv[3];

  if (arg1 && arg1.endsWith('.json')) {
    const inPath = path.resolve(arg1);
    const outPath = arg2 ? path.resolve(arg2) : path.join(root, 'data', 'witness_input.json');
    const input = loadLegacySingleFile(inPath);
    fs.writeFileSync(outPath, JSON.stringify(input, null, 2));
    console.log(`Wrote witness input to ${outPath}`);
    return;
  }

  const telemetryDir = arg1 ? path.resolve(arg1) : path.join(root, 'data', 'sample_telemetry');
  const witnessOutDir = arg2 ? path.resolve(arg2) : path.join(root, 'data', 'witnesses');
  generateFromDirectoryTree(telemetryDir, witnessOutDir);
}

main();
