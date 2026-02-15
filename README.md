# Intent-to-Auditable-Trust-Object (IATO)

IATO is a deterministic security assurance project focused on producing auditable trust artifacts from reproducible workflows.

## Repository layout

- `config/` — environment defaults, runtime/security settings, and setup helpers.
- `scripts/` — core research and orchestration scripts.
- `tests/` — validation code, schemas, and sample data.
- `docs/` — project documentation and notes.
- `docker/` — local container and observability stack assets.
- `ci/` — CI checks and environment validation scripts.
- `bin/` — archived/legacy helper files that are not part of the primary workflow.

## Getting started

1. Create the environment:
   - `conda env create -f environment.yml`
2. Activate it:
   - `conda activate testenv`
3. Run repository checks:
   - `bash ci/tools/run_all_checks.sh`

## Scope note

This repository contains both active code and historical artifacts. Prefer content under `config/`, `scripts/`, `tests/`, `docs/`, `docker/`, and `ci/tools/` for current workflows.
