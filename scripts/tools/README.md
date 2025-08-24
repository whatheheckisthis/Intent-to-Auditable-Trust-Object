# Utility Scripts

This directory contains standalone scripts designed to support project maintenance, diagnostics, and release workflows. Each script is self-contained and intended for manual use or integration with CI/CD systems.

## Available Scripts

| Script | Purpose |
|--------|---------|
| `cleanup_logs.sh` | Archive logs older than 7 days and delete stale archives |
| `tag_release.sh` | Create annotated Git tags and optionally push to remote |
| `generate_report.py` | Summarize recent Git activity, logs, and file sizes |
| `run_diagnostics.sh` | Check system setup: Python, Conda, memory, internet |
| `build_package.sh` | Clean and build Python packages using `pyproject.toml` |
| `check_licenses.sh` | Audit Python package licenses using `pip-licenses` |
| `clean_workspace.sh` | Remove Python cache, build artifacts, and temp files |

## Usage

Run any script from the root of the repository:

```bash
bash scripts/<script_name>.sh
