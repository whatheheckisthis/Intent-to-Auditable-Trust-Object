#!/usr/bin/env node
/**
 * Query on-chain verification result for a sequence number.
 * Env: RPC_URL, TELEMETRY_PROOF_ADDR
 */
const fs = require('fs');
const path = require('path');
const { ethers } = require('ethers');

async function main() {
  const root = path.resolve(__dirname, '..');
  const artifact = JSON.parse(fs.readFileSync(path.join(root, 'artifacts', 'TelemetryProof.json'), 'utf8'));

  const provider = new ethers.JsonRpcProvider(process.env.RPC_URL);
  const contract = new ethers.Contract(process.env.TELEMETRY_PROOF_ADDR, artifact.abi, provider);

  const seqNo = Number(process.argv[2] || process.env.SEQ_NO || 0);
  const status = await contract.results(seqNo);
  console.log(`Sequence ${seqNo} verification result: ${status}`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
