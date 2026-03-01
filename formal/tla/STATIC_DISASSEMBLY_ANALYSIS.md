# Static Disassembly Analysis for Branchless Reduction

This companion pack extends `dijkstra_branchless_assurance.tla` with **60 TLC scenario configs**, each configured for **10,000 scenarios** and intended for static compilation/disassembly checks.

## Why this exists

The TLA+ model proves algorithm-level path determinism, but two residual risks remain outside pure model checking:

1. **Compiler reintroduction**: optimizer rewrites arithmetic mask logic into conditional instructions.
2. **Microarchitectural speculation**: flag-producing compares may still influence speculative behavior.

This document addresses risk (1) with a reproducible static disassembly workflow that inspects **hex + mnemonic** output for forbidden instructions.

## Scenario corpus (archive status)

The previous 60-file scenario corpus under `formal/tla/scenarios/` was purged as part of repository cleanup. Use the retained baseline config `formal/tla/dijkstra_branchless_assurance.cfg` for reproducible checks.

## Disassembly procedure

Use `formal/tla/tools/static_disassembly_check.sh` with a branchless target function.

### Example (x86_64 host)

```bash
formal/tla/tools/static_disassembly_check.sh \
  formal/tla/tools/branchless_reduction_sample.c \
  branchless_reduce \
  x86_64 \
  cc
```

### Example (AArch64 cross toolchain)

```bash
formal/tla/tools/static_disassembly_check.sh \
  formal/tla/tools/branchless_reduction_sample.c \
  branchless_reduce \
  arm64 \
  aarch64-linux-gnu-gcc
```

The script emits function disassembly lines (hex + mnemonic) and fails if forbidden conditional instructions are present.

## Forbidden instruction policy

- **arm64**: reject `B.<cond>`, `CBZ/CBNZ`, `TBZ/TBNZ`, `CSEL` family.
- **x86_64**: reject `Jcc` and `CMOVcc`.

## Expected assurance interpretation

- TLC + these scenario configs provide algorithmic/state-space assurance over 10k-scenario runs.
- Static disassembly check provides compile-output assurance that branchless intent remains branchless in machine code.
- Microarchitectural timing/speculation validation remains a separate required artifact.


## ASL mapping constraints used for branchless validation

The branchless TLA transition is mapped to ARMv9/ASL-level intent as:

```
TLA⁺ State Variable     →   ASL / ARMv9 Construct
────────────────────────────────────────────────────────────────────
t ∈ [0, 2·M)            →   Xn register (64-bit unsigned)
mask = 0 − (t ≥ M)      →   SUBS X_tmp, X_t, X_M
                             NGC  X_mask, XZR
M AND mask              →   AND  X_sub, X_M, X_mask
t − (M AND mask)        →   SUB  X_result, X_t, X_sub
path_taken = constant   →   No B.xx / CBZ / CBNZ in instruction sequence
```

### Refinement obligation (recorded for proof artifacts)

```
∀ input T ∈ [0, M·R) :
    ASL_Execute(Montgomery_Branchless, T)
        refines TLA_Execute(Next_Branchless, T)
    ∧ instruction_trace(T₁) = instruction_trace(T₂)
```

A reference AArch64 assembly target following this sequence is provided at
`formal/tla/tools/branchless_reduction_arm64_asm.S` for downstream disassembly checks.
