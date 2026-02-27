#!/usr/bin/env python3
"""Validate required Python imports with actionable error messages."""

from __future__ import annotations

import importlib
import sys

REQUIRED_IMPORTS = {
    "boto3": "pip install boto3",
    "yaml": "pip install pyyaml",
    "requests": "pip install requests",
    "policyuniverse": "pip install policyuniverse",
}


def main() -> int:
    missing: list[tuple[str, str]] = []

    for module_name, install_hint in REQUIRED_IMPORTS.items():
        try:
            importlib.import_module(module_name)
            print(f"[verify_imports] ok: {module_name}")
        except Exception as exc:  # noqa: BLE001
            print(f"[verify_imports] missing: {module_name} ({exc})")
            missing.append((module_name, install_hint))

    if missing:
        print("[verify_imports] error: one or more required imports are unavailable")
        for module_name, install_hint in missing:
            print(f"  - {module_name}: {install_hint}")
        return 1

    print("[verify_imports] all required imports available")
    return 0


if __name__ == "__main__":
    sys.exit(main())
