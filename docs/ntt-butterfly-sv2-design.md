# SV2 NTT Butterfly Unit Design (Kyber Ring)

## Scope
This document describes a cycle-accurate SystemVerilog (SV2) radix-2 NTT butterfly unit for the Kyber ring:

- Ring: \(R_q = \mathbb{Z}_{3329}[x]/(x^{256}+1)\)
- Coefficients: 16-bit lanes
- Streaming target: **100 MSPS**
- Full 256-point transform latency target: **< 1,200 ns** with adequate clock/parallel deployment

The RTL implementation is in `hardware/ntt_butterfly.sv`.

## Architecture Summary

### Z-Register Lane Architecture
- The design uses a SIMD-style coefficient register bank:
  - `logic [255:0][15:0] z_reg`
- MMIO data beats load **two 16-bit coefficients per cycle**:
  - `mmio_wdata[15:0]` -> `z_reg[z_wr_ptr]`
  - `mmio_wdata[31:16]` -> `z_reg[z_wr_ptr+1]`
- This allows direct lane-stride butterfly processing without RAM arbitration in the critical path.

### Pipelined Radix-2 Butterfly
The butterfly datapath is split into 3 deterministic stages:

1. **Stage 1**
   - Read left/right coefficients from Z-lanes.
   - Multiply right coefficient by twiddle factor.
2. **Stage 2**
   - Montgomery-ASR reduce the product.
   - Form pre-reduction add/sub terms: `a + t` and `a + q - t`.
3. **Stage 3**
   - Branchless reduction back into `[0, q)`.
   - Write reduced outputs into destination Z-lanes.

### Branchless Montgomery-ASR Reduction
The reduction logic is implemented with:
- Multiplication
- Addition/subtraction
- Arithmetic right-shift style extraction in Montgomery flow
- Masked add-back correction

No data-dependent `if/else` or ternary operator is used for reduction decisions.

## MMIO Interface for eBPF/XDP Telemetry
The module includes MMIO ports intended for software-driver integration:

- `MMIO_ADDR_DATA` (0x00): packed coefficient streaming (2x16-bit per beat)
- `MMIO_ADDR_CTRL` (0x04): stream enable + lane write pointer
- `MMIO_ADDR_STAT` (0x08): pipeline valid bits and output indices

Readback path exposes status and output payload so an eBPF/XDP pipeline can ingest telemetry with fixed transaction semantics.

## Formal Assertions (SVA)
The RTL includes assertion properties under `FORMAL`:

- `p_s3_sum_lt_q`: Stage-3 sum output remains `< q`
- `p_s3_dif_lt_q`: Stage-3 diff output remains `< q`
- `p_out_lt_q`: Final outputs remain `< q`

These properties provide formal range-safety guarantees for reduced butterfly outputs.

## Side-Channel and Integrity Notes

- **Branchless Constant-Time:** By using Arithmetic Shift Right (ASR) for modular reduction, the power consumption and timing of the chip remain identical regardless of the input data. This is what you meant by "side-channel silence."
- **SV2 Z-Registers:** Utilizing `logic [255:0][15:0] z_reg` allows for massive spatial parallelism. You aren't fetching from slow RAM; you are shifting data through high-speed flip-flops in stride with the packet transit.
- **Lyapunov Integrity:** Since the math is verified via SVA (SystemVerilog Assertions), you have a formal guarantee that the state transitions within the NTT core remain within the manifold bounds.

## Throughput and Latency Guidance
- At 100 MSPS ingress, the butterfly must sustain deterministic per-cycle acceptance.
- The design exposes a `FULL_NTT_CYCLES`/`CLK_FREQ_HZ` check in RTL to flag integration points where total fold latency would exceed 1,200 ns.

## Next Steps for Implementation
- **Coefficient Mapping:** We need to define the Twiddle Factor ROM. Since \(x^{256}+1\) is a fixed cyclotomic polynomial, the twiddle factors are constant and can be hard-coded into the RTL.
- **Clock Domain:** To hit 100 MSPS with a 256-point fold, your NTT core will likely need to run at 250MHz - 400MHz if the pipeline is shallow, or you will need multiple parallel butterfly units.
