# Intent-to-Auditable-Trust-Object (IĀTŌ-V7)

Formal verification framework for FEAT_RME + TS/SCI non-interference on ARMv9-A.

## Architecture

```text
lean/          # Canonical Lean 4 formal verification package
├── IATO/V7/   # Core formal model
│   ├── Basic.lean
│   ├── Worker.lean
│   └── Architecture.lean
└── Test/      # Unit tests

workers/       # Legacy → IATO-V7 migration
├── legacy/    # Retained only if compatible/migratable
├── new/       # Migrated outputs
└── compatibility/ # Scan reports

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

## Legacy cleanup policy

Legacy artifacts are analyzed for compatibility against the canonical `lean/` package and purged when inadequate.
See:

- `docs/WORKER_COMPAT.md`
- `workers/legacy/README.md`


## Branch hygiene

Irrelevant legacy root folders are purged from main branch to keep the repository aligned with the current direction. See `docs/LEGACY_PURGE.md`.
