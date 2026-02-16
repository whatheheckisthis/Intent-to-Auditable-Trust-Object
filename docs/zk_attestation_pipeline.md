# High-Throughput Recursive zk-SNARK Prover Pipeline (100 MSPS)

## 1) Pipeline Overview

The attestation path is split into five hardware/software stages:

1. **Packet ingest (XDP/eBPF):** packet headers are normalized into an OSINT metadata tuple.
2. **MMIO bridge:** metadata is pushed into FPGA command registers.
3. **Parallel NTT core:** batched polynomial arithmetic for Groth16/Plonky2 arithmetization.
4. **Recursive aggregation circuit:** 1024 leaf packet proofs are folded into one epoch witness.
5. **Verifier export:** 256-bit witness attached to packet metadata and checkpointed per epoch.

At 100 MSPS, the design assumes decoupled buffering and multi-epoch overlap (epoch `k` proving while epoch `k+1` is ingesting).

## 2) Throughput and Latency Targets

- **Input rate:** 100 million samples/packets per second.
- **NTT backend throughput target:**
  \[
  \text{ops/s} \approx f_{clk} \times \text{lanes} \times \text{utilization}
  \]
  Example at 500 MHz, 8 lanes, 80% utilization: **3.2e9 modular ops/s**.
- **End-to-end latency target per batch:** **< 1200 ns** from packet-batch fence to witness availability.

### Nominal budget (example)

- eBPF metadata extraction + map write: 120 ns
- MMIO enqueue/dequeue doorbell: 160 ns
- NTT + MSM micro-pipeline: 680 ns
- Recursive fold + witness finalization: 180 ns
- Metadata append and return: 40 ns

Total: **1180 ns**.

## 3) Recursive Aggregation Logic

- Batch size per epoch: **1024** packet proofs.
- Aggregation primitive: Poseidon fold chain.
- Constraint: every leaf proof must present `proofValid = 1`.
- Witness emission: split fold digest into two 128-bit limbs (256-bit logical witness).

## 4) Files in this implementation

- `hardware/parallel_ntt_core.sv`
  - Parallel butterfly scheduler + pipelined Montgomery multiplier lanes.
  - Emits epoch witness (`witness_256`) and done pulse.
- `ebpf/osint_snark_bridge.bpf.c`
  - XDP program using `bpf_probe_read_kernel` for metadata snapshot.
  - Pushes packet metadata into FPGA MMIO shadow map.
  - Polls proof-ready status and appends 256-bit witness into XDP metadata area.
- `circuits/recursive_osint_aggregator.circom`
  - Recursive packet-proof fold circuit with `N=1024` leaf proofs.
- `formal/witness_integrity.als`
  - Alloy model proving forged packet leaves cannot appear in accepted recursive proofs within bounded window.

## 5) Integration Notes

- In production, eBPF should pair with a privileged user-space BAR relay to access actual PCIe MMIO.
- Twiddle tables and witness hash schedule in RTL are placeholders and should be generated from proving field parameters.
- For Plonky2-style recursion, replace witness fold arity and field modulus with Goldilocks-compatible constants.
