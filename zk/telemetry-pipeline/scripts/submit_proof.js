#!/usr/bin/env node
/**
 * Submit Groth16 proof + public inputs to TelemetryProof contract.
 * Env: RPC_URL, PRIVATE_KEY, TELEMETRY_PROOF_ADDR
 */
const fs = require('fs');
const path = require('path');
const { ethers } = require('ethers');

async function main() {
  const root = path.resolve(__dirname, '..');
  const contractArtifact = JSON.parse(fs.readFileSync(path.join(root, 'artifacts', 'TelemetryProof.json'), 'utf8'));
  const proof = JSON.parse(fs.readFileSync(path.join(root, 'artifacts', 'proofs', 'proof.json'), 'utf8'));
  const pub = JSON.parse(fs.readFileSync(path.join(root, 'artifacts', 'proofs', 'public.json'), 'utf8'));

  const provider = new ethers.JsonRpcProvider(process.env.RPC_URL);
  const wallet = new ethers.Wallet(process.env.PRIVATE_KEY, provider);
  const contract = new ethers.Contract(process.env.TELEMETRY_PROOF_ADDR, contractArtifact.abi, wallet);

  const a = proof.pi_a.slice(0, 2);
  const b = [proof.pi_b[0].slice().reverse(), proof.pi_b[1].slice().reverse()];
  const c = proof.pi_c.slice(0, 2);

  const epochId = Number(process.env.EPOCH_ID || 1);
  const seqNo = Number(process.env.SEQ_NO || Date.now());
  const witnessHash = ethers.keccak256(ethers.toUtf8Bytes(pub.join(',')));

  const tx = await contract.submitProof(epochId, seqNo, witnessHash, a, b, c, pub);
  const receipt = await tx.wait();
  console.log(`Proof submitted in tx: ${receipt.hash}`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
