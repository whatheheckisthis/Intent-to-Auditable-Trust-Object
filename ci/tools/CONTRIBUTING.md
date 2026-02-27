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

### Bootstrap behavior controls

The canonical bootstrap script supports optional strictness and worker package installation:

- `BOOTSTRAP_STRICT=1` (or `--strict`): fail immediately when dependency installation fails.
- `BOOTSTRAP_INSTALL_WORKER_PACKAGES=1` (or `--install-worker-packages`): ensure worker prerequisites (`git`, `curl`, `unzip`, `jq`) are installed before Python bootstrap.

These environment variables are honored when running `ci/tools/run_all_checks.sh`.

Additional container/shell integration controls:

- `BOOTSTRAP_INSTALL_CONTAINER_PACKAGES=1` (or `--install-container-packages`): try to install `docker` and `podman` commands on workers.
- `BOOTSTRAP_CONTAINER_RUNTIME=<docker|podman|auto>` (or `--container-runtime`): select preferred runtime for integration output.
- `BOOTSTRAP_SHELL=<bash|sh|auto>` (or `--shell`): select preferred shell for integration output.
- `BOOTSTRAP_RUNTIME_ENV_FILE=<path>`: output file for discovered shell/runtime exports (default: `ci/tools/.env.runtime`).

