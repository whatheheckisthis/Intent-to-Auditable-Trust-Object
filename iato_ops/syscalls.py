"""System-call-style runtime metadata helpers for low-level dispatcher integrations."""

from __future__ import annotations

import os
import platform
import time
from typing import Any


def collect_runtime_syscalls() -> dict[str, Any]:
    """Collect a minimal, JSON-serializable syscall/runtime metadata snapshot."""

    return {
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "cwd": os.getcwd(),
        "uname": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "monotonic_ns": time.monotonic_ns(),
        "time_ns": time.time_ns(),
    }
