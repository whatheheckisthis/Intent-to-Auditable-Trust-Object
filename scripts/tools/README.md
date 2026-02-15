# `scripts/tools` utilities

Standalone maintenance and diagnostics utilities used during development.

## Common scripts

- `cleanup_logs.sh` — rotates/removes old logs.
- `tag_release.sh` / `tag_release2.sh` — tag helpers.
- `generate_report.py` — project activity/report helper.
- `run_diagnostics.sh` — local system checks.
- `build_package.sh` — package build helper.
- `check_licenses.sh` — dependency license checks.
- `clean_workspace.sh` — clears temporary/build artifacts.

## Usage

Run from repository root:

```bash
bash scripts/tools/<script_name>.sh
```
