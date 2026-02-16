# High-Throughput Recursive zk-SNARK Prover Pipeline (100 MSPS)

## 1) Pipeline Overview

The attestation path is split into five hardware/software stages:

1. **Packet ingest (XDP/eBPF):** packet headers are normalized into an OSINT metadata tuple.
2. **MMIO bridge:** metadata is pushed into FPGA command registers.
3. **Parallel NTT core:** batched polynomial arithmetic for Groth16/Plonky2 arithmetization.
4. **Recursive aggregation circuit:** 1024 leaf packet proofs are folded into one epoch witness.
5. **Verifier export:** 256-bit witness attached to packet metadata and checkpointed per epoch.

At 100 MSPS, the design assumes decoupled buffering and multi-epoch overlap (epoch `k` proving while epoch `k+1` is ingesting).

## 2) Ring and NTT Butterfly

- Target ring: Kyber-friendly modulus `q = 3329`.
- Butterfly primitive: radix-2 `a' = a + b\omega`, `b' = a - b\omega (mod q)`.
- Montgomery-ASR reduction:
  - `u = (t * q^{-1}) mod 2^16`
  - `r = (t - u*q) >>> 16`
  - reduce `r` into `[0, q)`.
- Side-channel silence: coefficients remain in streaming Z-register lanes and avoid data-dependent BRAM lookups.

## 3) Lane Mapping and Latency Budget (N=256)

- Polynomial degree: 256 coefficients (`LOGN=8`).
- Lane map: coefficient index `k` is assigned to lane `k % LANES` each scheduler batch.
- Default lane count: `LANES=16`.
- Stage work: `N/2 = 128` butterflies per stage, `8` batches/stage at 16 lanes.
- 8 stages + pipeline slack => estimated fold latency `< 1200ns` at 500 MHz (`CLOCK_NS=2`).

## 4) Lyapunov Manifold Witness Routine

`telemetry_to_stability_witness()` maps each packet telemetry tuple into a bounded Lyapunov state `x` and computes:

- `V0 = x^T P x`
- `VN = V0 * exp(-alpha * steps)` with `steps = 1e8` by default
- `margin = clamp(V0 - VN, 0, 1)`

The resulting witness is a 256-bit value serialized as four 64-bit words (`V0`, `VN`, `margin`, `steps`).

## 5) ESCALATE() XDP Behavior

- XDP pushes packet metadata to the FPGA MMIO shadow map.
- If witness status marks verification failure (`FPGA_WITNESS_FAIL_MASK`):
  - Packet is dropped immediately.
  - A high-priority `perf_event` alert is emitted (`escalate_alerts` map).
  - Program returns `XDP_ABORTED` to force fail-closed behavior.

> Note: eBPF/XDP cannot legally invoke a direct kernel panic from datapath context. The implementation uses immediate drop/abort + high-priority alerting as the safe equivalent.

## 6) Formal Specification (TLA+)

Invariant:

`verify(pi, witness) = TRUE <=> all transitions remain inside manifold bounds`.

Encoded in `formal/tla/montgomery_reduction_invariant.tla` as theorem `VerifyIffTransitionsBounded`.

## 7) Files in this implementation

- `hardware/ntt_butterfly.sv`
  - Radix-2 Kyber butterfly with Montgomery-ASR reduction.
- `hardware/parallel_ntt_core.sv`
  - N=256 lane scheduler and 256-bit witness fold output.
- `ebpf/osint_snark_bridge.bpf.c`
  - XDP DMA-bridge model with `ESCALATE()` alert/drop behavior.
- `ebpf/telemetry_lyapunov_witness.c`
  - Functional C routine from telemetry tuple to 256-bit stability witness.
- `formal/tla/montgomery_reduction_invariant.tla`
  - IFF verification invariant over bounded transitions.
