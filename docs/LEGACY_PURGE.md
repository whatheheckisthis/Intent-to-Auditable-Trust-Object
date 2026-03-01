# Legacy Purge Log

To keep the main branch aligned with the current Lean-first direction, irrelevant legacy folders were removed from repository root.

## Purged root folders

- `.vscode/`
- `bin/`
- `changelog/`
- `ci/`
- `circuits/`
- `compat/`
- `docker/`
- `ebpf/`
- `el2/`
- `formal/`
- `grafana/`
- `hardware/`
- `iato_ops/`
- `include/`
- `infrastructure/`
- `ipfs/`
- `kernel/`
- `monitoring/`
- `nfc/`
- `prometheus/`
- `proxy/`
- `src/`
- `tests/`
- `zk/`

## Kept root direction

Main branch now centers on:

- `lean/` (canonical formal package)
- `docs/` (architecture and compatibility docs)
- `workers/` (legacy/new migration areas)
- `scripts/` (build and migration tooling)
- `data/` (sample migration inputs)
