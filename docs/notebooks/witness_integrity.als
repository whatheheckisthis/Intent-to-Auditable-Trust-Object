// Witness integrity model for recursive zk-SNARK packet attestation.
// Goal: show forged packet cannot produce a valid recursive proof within
// bounded attestation window.

open util/integer

one sig EpochCfg {
  // Window in microseconds for accepting packet->proof linkage.
  attestationWindowUs: one Int,
  maxBatch: one Int
}

sig Packet {
  id: one Int,
  telemetryHash: one Int,
  recvTimeUs: one Int,
  forged: one Bool
}

sig LeafProof {
  p: one Packet,
  valid: one Bool,
  genTimeUs: one Int
}

sig RecursiveProof {
  leaves: set LeafProof,
  witness: one Int,
  sealTimeUs: one Int,
  accepted: one Bool
}

pred LeafValidity(lp: LeafProof) {
  lp.valid = True implies lp.p.forged = False
}

pred InWindow(lp: LeafProof, rp: RecursiveProof) {
  rp.sealTimeUs - lp.p.recvTimeUs <= EpochCfg.attestationWindowUs
}

pred RecursiveAcceptsOnlyValid(rp: RecursiveProof) {
  rp.accepted = True implies
    (all lp: rp.leaves | lp.valid = True and InWindow(lp, rp))
}

// Forge resistance property:
// no accepted recursive proof may contain forged packet leaves.
assert NoForgedPacketCanBeAccepted {
  all rp: RecursiveProof |
    RecursiveAcceptsOnlyValid(rp) and
    (all lp: rp.leaves | LeafValidity(lp))
    implies
    (all lp: rp.leaves | lp.p.forged = False)
}

// Bounded throughput sanity: batch size up to configured maximum.
assert BatchBounded {
  all rp: RecursiveProof | #rp.leaves <= EpochCfg.maxBatch
}

pred ExampleCfg {
  EpochCfg.attestationWindowUs = 1200
  EpochCfg.maxBatch = 1024
}

check NoForgedPacketCanBeAccepted for 10 but 16 Int
check BatchBounded for 10 but 16 Int
run ExampleCfg for 10 but 16 Int
