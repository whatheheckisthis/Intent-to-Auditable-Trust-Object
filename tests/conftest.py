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


import os
import pytest


@pytest.fixture(scope="session")
def hw_mode() -> bool:
    return os.environ.get("IATO_HW_MODE", "0") == "1"


class _SimSmcInterface:
    def __init__(self):
        from src.el2_validator import El2CredentialValidator
        self.validator = El2CredentialValidator()
        self.ste = {}

    def provision(self, stream_id, raw_credential):
        try:
            cred = self.validator.validate(raw=raw_credential, stream_id=stream_id)
        except Exception:
            return 1
        self.ste[stream_id] = 1 | ((cred.permissions & 0x3) << 6)
        return 0


class _SimSmmuReader:
    def __init__(self, smc):
        self.smc = smc

    def read_ste_word0(self, stream_id):
        return self.smc.ste.get(stream_id, 0)


@pytest.fixture(scope="session")
def smc_interface(hw_mode):
    _ = hw_mode
    return _SimSmcInterface()


@pytest.fixture(scope="session")
def smmu_reader(hw_mode, smc_interface):
    _ = hw_mode
    return _SimSmmuReader(smc_interface)
