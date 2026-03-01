# IATO-V7 Architecture

## Formal Verification Layers

```text
DepSet Lattice (⊆, ⊔, ⊥) ──┐
        ↓                   │ TS/SCI Type System
Worker Model ───────────────┼── Non-interference
        ↓                   │    Fundamental Theorem
FEAT_RME Axioms ────────────┘
```

## Migration Pipeline

```text
legacy_workers.csv ──[scan]──> compatibility/scan_report.json
                                      ↓
                               [migrate] ──> workers/new/
```
