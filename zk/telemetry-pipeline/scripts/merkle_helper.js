#!/usr/bin/env node
/**
 * Build SHA-256 Merkle root for telemetry micro-batches.
 * Usage: node scripts/merkle_helper.js data/sample_telemetry.json
 */
const fs = require('fs');
const crypto = require('crypto');
const path = require('path');

function hashLeaf(v) {
  return crypto.createHash('sha256').update(Buffer.from(String(v))).digest();
}

function hashNode(left, right) {
  return crypto.createHash('sha256').update(Buffer.concat([left, right])).digest();
}

function merkleRoot(values) {
  let level = values.map(hashLeaf);
  while (level.length > 1) {
    if (level.length % 2 === 1) level.push(level[level.length - 1]);
    const next = [];
    for (let i = 0; i < level.length; i += 2) {
      next.push(hashNode(level[i], level[i + 1]));
    }
    level = next;
  }
  return `0x${level[0].toString('hex')}`;
}

const root = path.resolve(__dirname, '..');
const input = process.argv[2] || path.join(root, 'data', 'sample_telemetry.json');
const t = JSON.parse(fs.readFileSync(input, 'utf8'));

console.log(JSON.stringify({
  lotARoot: merkleRoot(t.lotA),
  lotBRoot: merkleRoot(t.lotB),
  fullRoot: merkleRoot(t.lotA.concat(t.lotB))
}, null, 2));
