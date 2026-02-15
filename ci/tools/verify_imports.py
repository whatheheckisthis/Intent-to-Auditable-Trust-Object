#!/usr/bin/env python3
from pathlib import Path
import runpy

repo_root = Path(__file__).resolve().parents[2]
runpy.run_path(repo_root / "scripts/cli/architecture/verify_imports.py", run_name="__main__")
