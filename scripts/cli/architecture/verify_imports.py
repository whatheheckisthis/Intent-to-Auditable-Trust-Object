#!/usr/bin/env python3
"""Verify that critical Python dependencies import cleanly."""

import importlib
import sys

REQUIRED_IMPORTS = ["numpy", "pytest", "flake8"]


def main() -> int:
    failures: list[str] = []
    for module_name in REQUIRED_IMPORTS:
        try:
            importlib.import_module(module_name)
            print(f"OK: {module_name}")
        except ImportError:
            print(f"FAIL: {module_name}")
            failures.append(module_name)

    if failures:
        print(f"Missing imports: {', '.join(failures)}")
        return 1

    print("All critical packages imported successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
