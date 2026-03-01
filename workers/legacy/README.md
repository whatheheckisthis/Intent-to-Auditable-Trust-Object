# Legacy Workers

This directory contains legacy or superseded worker-related artifacts.

## Compatibility triage

| Legacy source | Status | Action |
|---|---|---|
| Historical standalone Lean package (`IATO_V7/`) | Inadequate for current layout (duplicate package roots, duplicate CI/build metadata) | Purged from repository root lineage and replaced by canonical `lean/` package |
| Legacy worker runtime snippets | Partial compatibility | Keep under `workers/legacy/` until migrated |
| New migration outputs | Compatible target | Write to `workers/new/` |

## Migration policy

1. Scan legacy workers using `scripts/scan_workers.py`.
2. Record compatibility output in `workers/compatibility/scan_report.json`.
3. Promote migrated artifacts only into `workers/new/`.
