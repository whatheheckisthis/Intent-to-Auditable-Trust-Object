# Intent-to-Auditable-Trust-Object (IĀTŌ-V7)

Formal verification framework for FEAT_RME + TS/SCI non-interference on ARMv9-A.

## Architecture

```text
lean/          # Lean 4 formal verification package
├── IATO/V7/   # Core formal model
│   ├── Basic.lean
│   ├── Worker.lean
│   └── Architecture.lean
└── Test/      # Unit tests

workers/       # Legacy → IATO-V7 migration
├── legacy/
├── new/
└── compatibility/

docs/          # Specifications + architecture
scripts/       # Automation (scan, migrate, build)
data/          # Sample workers + reference architecture
```

## Quick Start

```bash
./scripts/lake_build.sh
python3 scripts/scan_workers.py data/legacy_workers.csv
./scripts/migrate.sh
```
