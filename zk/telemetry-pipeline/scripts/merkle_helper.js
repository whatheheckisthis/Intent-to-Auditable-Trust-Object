#!/usr/bin/env node
/**
 * Build keccak256 Merkle roots for telemetry witness batches.
 * Usage:
 *   node scripts/merkle_helper.js
 *   node scripts/merkle_helper.js data/witnesses
 */
const fs = require('fs');
const path = require('path');
const { ethers } = require('ethers');

function hashLeaf(value) {
  return ethers.keccak256(ethers.solidityPacked(['uint256'], [BigInt(value)]));
}

function hashNode(left, right) {
  return ethers.keccak256(ethers.solidityPacked(['bytes32', 'bytes32'], [left, right]));
}

function merkleRoot(values) {
  if (values.length === 0) {
    throw new Error('Cannot build a Merkle root from an empty set of values');
  }

  let level = values.map((v) => hashLeaf(v));
  while (level.length > 1) {
    if (level.length % 2 === 1) level.push(level[level.length - 1]);
    const next = [];
    for (let i = 0; i < level.length; i += 2) {
      next.push(hashNode(level[i], level[i + 1]));
    }
    level = next;
  }
  return level[0];
}

function getWitnessFiles(witnessDir) {
  return fs.readdirSync(witnessDir)
    .filter((name) => /^witness_batch_\d+\.json$/.test(name))
    .sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
}

function main() {
  const root = path.resolve(__dirname, '..');
  const witnessDir = process.argv[2] ? path.resolve(process.argv[2]) : path.join(root, 'data', 'witnesses');
  const outFile = process.argv[3] ? path.resolve(process.argv[3]) : path.join(witnessDir, 'merkle_roots.json');

  const witnessFiles = getWitnessFiles(witnessDir);
  if (witnessFiles.length === 0) {
    throw new Error(`No witness_batch_*.json files found in ${witnessDir}`);
  }

  const roots = {};
  const allWitnessValues = [];

  for (const file of witnessFiles) {
    const payload = JSON.parse(fs.readFileSync(path.join(witnessDir, file), 'utf8'));
    if (!Array.isArray(payload.witness)) {
      throw new Error(`${file} must contain a witness array`);
    }
    roots[file.replace('.json', '')] = merkleRoot(payload.witness);
    allWitnessValues.push(...payload.witness);
  }

  const result = {
    batchRoots: roots,
    combinedRoot: merkleRoot(allWitnessValues),
  };

  fs.writeFileSync(outFile, JSON.stringify(result, null, 2));
  console.log(JSON.stringify(result, null, 2));
}

main();
