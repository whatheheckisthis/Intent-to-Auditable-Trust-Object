pragma circom 2.1.6;

include "circomlib/circuits/comparators.circom";
include "circomlib/circuits/poseidon.circom";
include "circomlib/circuits/bitify.circom";

// Folded telemetry verifier circuit.
// - witness: 16-element folded telemetry vector (mod 3329) per micro-batch.
// - witnessBits: 512-bit folded witness emitted by SVE2/FPGA accumulator.
// - midpoint: Gaussian-seeded expected midpoint for integer divergence checks.
// - allowedDivergence: tolerated divergence around midpoint.
// - microBatchRoots: optional micro-batch roots that can be hashed into a combined root.
// - claimedCombinedRoot: public commitment used on-chain and in off-chain logs.
//
// The circuit exposes `valid` as a public output flag (1=true, 0=false).
template TelemetryFolding(batchCount, witnessWidth) {
    signal input witness[witnessWidth];
    signal input witnessBits[512];
    signal input midpoint;
    signal input allowedDivergence;
    signal input microBatchRoots[batchCount];
    signal input claimedCombinedRoot;

    signal output valid;
    signal output foldedWeight;
    signal output combinedRoot;

    // Constrain witness coefficients for micro-batch usage:
    // - each coefficient is represented in <= 12 bits
    // - each coefficient is < 3329
    // - each coefficient parity is linked to witnessBits[i]
    var i;
    for (i = 0; i < witnessWidth; i++) {
        component coeffBits = Num2Bits(12);
        coeffBits.in <== witness[i];

        component coeffLtMod = LessThan(13);
        coeffLtMod.in[0] <== witness[i];
        coeffLtMod.in[1] <== 3329;
        coeffLtMod.out === 1;

        witnessBits[i] === coeffBits.out[0];
    }

    // Constrain each witness bit and fold into Hamming weight.
    signal runningWeight[513];
    runningWeight[0] <== 0;

    for (i = 0; i < 512; i++) {
        witnessBits[i] * (witnessBits[i] - 1) === 0;
        runningWeight[i + 1] <== runningWeight[i] + witnessBits[i];
    }
    foldedWeight <== runningWeight[512];

    // Divergence check: |foldedWeight - midpoint| <= allowedDivergence
    component lt = LessThan(11); // values are <= 512, midpoint <= 1023
    lt.in[0] <== foldedWeight;
    lt.in[1] <== midpoint;

    signal diff;
    signal diffPos;
    signal diffNeg;
    diffPos <== foldedWeight - midpoint;
    diffNeg <== midpoint - foldedWeight;
    diff <== lt.out * diffNeg + (1 - lt.out) * diffPos;

    component leq = LessEqThan(11);
    leq.in[0] <== diff;
    leq.in[1] <== allowedDivergence;

    // Modular root fold for micro-batch expansion.
    // combinedRoot = Poseidon(Poseidon(...Poseidon(0, root_0), root_1)...)
    signal runningRoot[batchCount + 1];
    runningRoot[0] <== 0;

    for (i = 0; i < batchCount; i++) {
        component p = Poseidon(2);
        p.inputs[0] <== runningRoot[i];
        p.inputs[1] <== microBatchRoots[i];
        runningRoot[i + 1] <== p.out;
    }

    combinedRoot <== runningRoot[batchCount];

    component rootEq = IsZero();
    rootEq.in <== combinedRoot - claimedCombinedRoot;

    valid <== leq.out * rootEq.out;
}

component main = TelemetryFolding(2, 16);
