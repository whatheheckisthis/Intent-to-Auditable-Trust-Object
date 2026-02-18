#!/usr/bin/env node
/**
 * Deploy TelemetryProof contract using ethers.js.
 * Expects env vars: RPC_URL, PRIVATE_KEY.
 */
const fs = require('fs');
const path = require('path');
const { ethers } = require('ethers');

async function main() {
  const root = path.resolve(__dirname, '..');
  const artifact = JSON.parse(fs.readFileSync(path.join(root, 'artifacts', 'TelemetryProof.json'), 'utf8'));

  const provider = new ethers.JsonRpcProvider(process.env.RPC_URL);
  const wallet = new ethers.Wallet(process.env.PRIVATE_KEY, provider);
  const factory = new ethers.ContractFactory(artifact.abi, artifact.bytecode, wallet);

  const verifierAddress = process.env.GROTH16_VERIFIER;
  const contract = await factory.deploy(verifierAddress);
  await contract.waitForDeployment();
  console.log(`TelemetryProof deployed at: ${await contract.getAddress()}`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
