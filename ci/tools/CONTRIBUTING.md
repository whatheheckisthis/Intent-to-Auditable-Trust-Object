### CI Tooling Architecture

CI validation is now organized around canonical scripts in:

- `scripts/cli/architecture/`

The `ci/tools/` scripts are maintained as compatibility entry points and wrappers.

| Entry Point | Canonical Script | Purpose |
|--------|-------------|---|
| `ci/tools/verify_environment.sh` | `scripts/cli/architecture/verify_environment.sh` | Validates system dependencies and Conda availability |
| `ci/tools/verify_imports.py` | `scripts/cli/architecture/verify_imports.py` | Verifies critical Python imports |
| `ci/tools/lint.sh` | `scripts/cli/architecture/lint.sh` | Runs `flake8` lint checks |
| `ci/tools/bootstrap_env.sh` | `scripts/cli/architecture/bootstrap_env.sh` | Installs baseline Python packages in `testenv` |
| `ci/tools/run_all_checks.sh` | `scripts/cli/architecture/run_all_checks.sh` | Aggregates environment + import + lint checks |

### Changelog Scripts

Changelog utilities are now centralized in:

- `changelog/scripts/`

Legacy files under `bin/` now forward to this location for compatibility.
