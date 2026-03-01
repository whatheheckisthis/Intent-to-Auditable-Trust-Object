# Worker Compatibility

This document tracks whether legacy worker and formal artifacts are compatible with the canonical IATO-V7 layout.

## Current compatibility findings

| Area | Compatible? | Notes |
|---|---|---|
| Canonical Lean package in `lean/` | Yes | Current source of truth for formal model and tests |
| Legacy duplicate Lean package (`IATO_V7/`) | No | Removed as obsolete duplicate with conflicting package roots and CI files |
| Worker migration scaffolding (`workers/`, `data/`, `scripts/`) | Yes | Scanner and migration scripts available |

## Adequacy criteria

An artifact is considered *adequate* only if it:

- Builds or integrates with current package layout (`lean/`).
- Does not duplicate top-level package/build metadata.
- Uses maintained import paths and active workflow scripts.

Artifacts failing these criteria are purged from the active branch.
