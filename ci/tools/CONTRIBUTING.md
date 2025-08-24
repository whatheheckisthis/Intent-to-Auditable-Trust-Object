

### Conda Environment Setup

The project uses GitHub Actions for CI with Conda environments managed via `setup-miniconda@v3`.

- `miniforge-variant` is set to `Miniforge3`
- `miniforge-version` is pinned (e.g., `23.3.1-0`) for reproducibility
- Dependencies are defined in `environment.yml`

### Validation Scripts in `ci/`

The `ci/` directory contains standalone scripts to verify the environment and enforce quality checks:

| Script | Description |
|--------|-------------|
| `verify_environment.sh` | Ensures required CLI tools are installed (`openssl`, `git`, `pandoc`, `python3`), checks Conda installation and environment activation |
| `verify_imports.py` | Checks that critical Python packages (`numpy`, `flake8`, `pytest`) can be successfully imported |
| `lint.sh` | Runs `flake8` to enforce Python style and detect syntax issues |
| `run_all_checks.sh` | Orchestrates all checks above in a single CI step |
| `init_environment.sh` | (Legacy) Early environment setup script, now replaced by the newer modular approach |

### CI Workflow Integration

These checks are integrated into GitHub Actions using the following step:

```yaml
- name: CI environment check
  shell: bash -l {0}
  run: |
    bash ci/run_all_checks.sh
