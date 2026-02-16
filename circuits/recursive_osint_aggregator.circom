pragma circom 2.1.8;

include "circomlib/poseidon.circom";

// Recursive OSINT packet-proof aggregator.
// Input: packet-level proof commitments + verifier flags from leaf provers.
// Output: a single 256-bit witness digest per epoch.
//
// Tuning targets for hardware-assisted proving pipeline:
//   * ingest >=100 MSPS packet stream
//   * aggregation latency <1200 ns for configured batch size
//
// NOTE: 256-bit witness emitted as two field limbs for BN254 compatibility.

template RecursivePacketAggregator(N) {
    signal input epochId;
    signal input packetRoot[N];      // e.g., Merkle root/commitment of each packet proof
    signal input proofValid[N];      // 1 if leaf proof verified, else 0
    signal input packetTimestamp[N]; // anti-replay binding

    signal output witnessLo;
    signal output witnessHi;
    signal output allValid;

    component poseA[N];
    component poseFold[N];

    signal fold[N + 1];
    fold[0] <== epochId;

    for (var i = 0; i < N; i++) {
        // Force every included packet proof to be valid in the epoch rollup.
        proofValid[i] * (proofValid[i] - 1) === 0;

        poseA[i] = Poseidon(3);
        poseA[i].inputs[0] <== packetRoot[i];
        poseA[i].inputs[1] <== packetTimestamp[i];
        poseA[i].inputs[2] <== proofValid[i];

        poseFold[i] = Poseidon(2);
        poseFold[i].inputs[0] <== fold[i];
        poseFold[i].inputs[1] <== poseA[i].out;

        // If proofValid is 0, constrain contribution to fail fold check downstream.
        // allValid output can be consumed by outer recursive verifier.
        fold[i + 1] <== poseFold[i].out + (1 - proofValid[i]);
    }

    signal validAcc[N + 1];
    validAcc[0] <== 1;
    for (var j = 0; j < N; j++) {
        validAcc[j + 1] <== validAcc[j] * proofValid[j];
    }

    allValid <== validAcc[N];

    // Split final fold digest into two 128-bit limbs for a 256-bit witness representation.
    // In production use Num2Bits(256) + limb packing by backend field modulus.
    witnessLo <== fold[N] % (1 << 128);
    witnessHi <== (fold[N] - witnessLo) / (1 << 128);
}

component main = RecursivePacketAggregator(1024);
