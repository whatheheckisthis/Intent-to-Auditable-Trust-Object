"""Pytest configuration for resilient local test collection."""

from __future__ import annotations

import importlib
import importlib.machinery
import sys
from pathlib import Path

pytest_plugins = ["src.hw_journal_pytest_plugin"]

# Ensure the repository root is importable during pytest collection.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _has_cocotb_runtime() -> bool:
    try:
        importlib.import_module("cocotb.clock")
        return True
    except Exception:
        return False


def pytest_ignore_collect(collection_path, config):
    """Skip cocotb hardware tests when cocotb runtime helpers are unavailable."""
    normalized = str(collection_path).replace('\\', '/')
    if '/tests/cocotb/' in normalized or normalized.endswith('/tests/cocotb'):
        return not _has_cocotb_runtime()
    return False


def pytest_configure(config):
    """Repair lightweight stubs that break importlib-based checks."""
    module = sys.modules.get("requests")
    if module is not None and getattr(module, "__spec__", None) is None:
        module.__spec__ = importlib.machinery.ModuleSpec("requests", loader=None)
