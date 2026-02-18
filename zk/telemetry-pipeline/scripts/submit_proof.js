#!/usr/bin/env node
/**
 * Submit Groth16 proof + public inputs to TelemetryProof contract.
 *
 * Env:
 * - RPC_URL, PRIVATE_KEY, TELEMETRY_PROOF_ADDR
 * - Optional: EPOCH_ID, SEQ_NO
 * - Optional hash inputs: WITNESS_HASH or MERKLE_ROOT
 *
 * Args:
 *   [proof_json] [public_json]
 */
const fs = require('fs');
const path = require('path');
const { ethers } = require('ethers');

function resolveWitnessCommitment(pubSignals) {
  if (process.env.WITNESS_HASH) return process.env.WITNESS_HASH;
  if (process.env.MERKLE_ROOT) return process.env.MERKLE_ROOT;
  return ethers.keccak256(ethers.toUtf8Bytes(pubSignals.join(',')));
}

async function main() {
  const root = path.resolve(__dirname, '..');
  const contractArtifact = JSON.parse(fs.readFileSync(path.join(root, 'artifacts', 'TelemetryProof.json'), 'utf8'));

  const proofPath = process.argv[2] || path.join(root, 'artifacts', 'proofs', 'proof.json');
  const publicPath = process.argv[3] || path.join(root, 'artifacts', 'proofs', 'public.json');

  const proof = JSON.parse(fs.readFileSync(proofPath, 'utf8'));
  const pub = JSON.parse(fs.readFileSync(publicPath, 'utf8'));

  const provider = new ethers.JsonRpcProvider(process.env.RPC_URL);
  const wallet = new ethers.Wallet(process.env.PRIVATE_KEY, provider);
  const contract = new ethers.Contract(process.env.TELEMETRY_PROOF_ADDR, contractArtifact.abi, wallet);

  const a = proof.pi_a.slice(0, 2);
  const b = [proof.pi_b[0].slice().reverse(), proof.pi_b[1].slice().reverse()];
  const c = proof.pi_c.slice(0, 2);

  const epochId = Number(process.env.EPOCH_ID || 1);
  const seqNo = Number(process.env.SEQ_NO || Date.now());
  const commitment = resolveWitnessCommitment(pub);

  const tx = await contract.submitProof(epochId, seqNo, commitment, a, b, c, pub);
  const receipt = await tx.wait();
  console.log(`Proof submitted in tx: ${receipt.hash}`);
  console.log(`Commitment used: ${commitment}`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
