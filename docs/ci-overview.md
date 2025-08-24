### Continuous Integration (CI) Overview

This document outlines the CI strategy and implementation for the `ops-utilities` repository. The goal is to ensure every push or pull request is automatically tested, validated, and linted in a reproducible, minimal, and secure runtime environment.

The CI system here supports reliability, maintainability, and onboarding transparency, especially for early-career contributors or changelog analysts working in operational or DevOps-adjacent roles.


CI Goals

The CI system is designed to:

- Ensure code consistency across environments (Linux-based workflows)
- Validate environment health (conda, dependencies, tools)
- Reduce contributor overhead by preempting runtime failures
- Serve as a template for future automation-first Bash-based utilities
- Build a workflow-first culture* that treats CI scripts as documentation of readiness

This makes the CI system not only a technical checkpoint but also a readable interface for understanding how the repo is structured.


Workflow Entry Point

The core GitHub Actions workflow is defined in:

```

`.github/workflows/python-package-conda.yml`

````

This workflow is triggered on each `push`. In future stages, it can be adapted to run only on `pull_request`, `main`, or `release/*` branches, depending on team scaling.



Conda-Based Environment Strategy

This project uses **Miniforge3** (a lightweight conda-forge-first variant) to ensure reproducibility and fast setup.

Key benefits:

- Keeps all CI setup within one unified `environment.yml`
- Avoids pip-only or OS-specific installation quirks
- Encourages explicit dependency declaration (`flake8`, `pytest`, `numpy`, etc.)

```yaml
with:
  `miniforge-variant: Miniforge3`
  `miniforge-version: 23.3.1-0  # Pinned for reproducibility`
  `python-version: 3.10`
  `environment-file: environment.yml`
  `activate-environment: testenv`
````

> Using `environment-file` is preferred over `conda install` inline calls for version tracking, caching, and diff readability in Git.

---

## Validation Steps

1. Tooling & Runtime Validation**

The first CI step checks:

* Whether essential binaries like `openssl`, `git`, `pandoc`, and `python3` exist
* Whether Conda is initialized
* Whether the expected Conda environment (`testenv`) is active

Script: `ci/verify_environment.sh`

2. Import Check for Critical Dependencies**

Before testing, the CI runs a small Python script to check for:

* Package importability (e.g., `numpy`, `flake8`, `pytest`)
* Detects subtle Conda installation failures or incomplete environments

Script: `ci/verify_imports.py`



3. Linting the Codebase**

Linting is enforced using `flake8`, with rules pinned in `.flake8`. It flags:

* Syntax issues
* Complexity and style violations
* Non-PEP8 code patterns

Script: `ci/lint.sh`

---

### 4. **Aggregated Check Runner**

For convenience and modularity, `ci/run_all_checks.sh` acts as a single entry point to all validation scripts. This allows future reuse outside of GitHub Actions (e.g., pre-commit checks or local dry runs).

📁 Script: `ci/run_all_checks.sh`

---

### 5. **Pytest Unit Testing**

Any test files located in `tests/` (e.g., `test_env_checks.py`) will be automatically discovered and executed.

```yaml
- name: Test with pytest
  shell: bash -l {0}
  run: |
    pytest
```

---

## Example Use Cases

| Use Case                                              | CI Component                     |
| ----------------------------------------------------- | -------------------------------- |
| Contributor pushes malformed Python code              | Fails at `lint.sh`               |
| Contributor forgets to update `environment.yml`       | Fails at `verify_imports.py`     |
| Repo is cloned into a broken Docker or VM environment | Fails at `verify_environment.sh` |
| New dependency added but not installed in CI          | Import check fails               |
| Refactor breaks unit logic                            | `pytest` fails                   |

---

## File Structure

```
ci/
├── lint.sh                # Codebase style enforcement
├── run_all_checks.sh      # Aggregates all validations
├── verify_environment.sh  # Checks binaries + Conda environment
├── verify_imports.py      # Checks critical Python imports

.github/workflows/
└── python-package-conda.yml

environment.yml            # Declarative Conda setup
.flake8                    # Linting ruleset
```

---

## Best Practices

* **Modularity**: Each check is standalone and testable locally.
* **Reusability**: `ci/` scripts are compatible with local dev, not just CI.
* **Fail Fast**: Environment problems show up immediately.
* **Documentation as CI**: Scripts are clear enough to double as onboarding guides.
* **Minimalism**: Avoids unnecessary Docker complexity; relies on native Unix + Conda.

---

## Future Roadmap

* [ ] Add `tests/` with unit and integration coverage
* [ ] Add badge support (CI status, coverage)
* [ ] Integrate secret scanning and shellcheck linters
* [ ] Enable PR-only workflows with branch protections
* [ ] Create reusable composite GitHub Action from these steps

---

## References

* [setup-miniconda GitHub Action](https://github.com/conda-incubator/setup-miniconda)
* [Conda Forge: Miniforge](https://github.com/conda-forge/miniforge)
* [Flake8 Documentation](https://flake8.pycqa.org/)
* [Issue #8: Conda Environment Verification](https://github.com/whatheheckisthis/ops-utilities/issues/8)



