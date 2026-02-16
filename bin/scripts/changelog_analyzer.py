#!/usr/bin/env python3
from pathlib import Path
import runpy
import sys

repo_root = Path(__file__).resolve().parents[1]
script_path = repo_root / "changelog/scripts/changelog_analyzer.py"
sys.argv[0] = str(script_path)
runpy.run_path(script_path, run_name="__main__")
