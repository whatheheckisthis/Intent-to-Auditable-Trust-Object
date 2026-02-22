from __future__ import annotations

import fcntl
import os
import threading
import time
from typing import Protocol

from src.hw_journal import get_hw_journal


class El2TimerBackend(Protocol):
    def arm(self, interval_ns: int) -> None: ...
    def disarm(self) -> None: ...
    def wait_for_expiry(self) -> None: ...


class ThreadingTimerBackend:
    def __init__(self):
        self._event = threading.Event()
        self._deadline = 0.0

    def arm(self, interval_ns: int) -> None:
        self._deadline = time.monotonic() + (interval_ns / 1_000_000_000)
        self._event.clear()
        get_hw_journal().record("cnthp", "timer_armed", data={"interval_ns": interval_ns, "backend": "threading"})

    def disarm(self) -> None:
        self._event.set()
        get_hw_journal().record("cnthp", "timer_disarmed", data={})

    def wait_for_expiry(self) -> None:
        start = time.monotonic_ns()
        timeout = max(0, self._deadline - time.monotonic())
        self._event.wait(timeout)
        get_hw_journal().record("cnthp", "timer_fired", data={"elapsed_ns": time.monotonic_ns() - start, "backend": "threading"})


class CnthpTimerBackend:
    TIMER_DEV = "/dev/iato-el2-timer"
    IOCTL_TIMER_ARM = 0xC0084101
    IOCTL_TIMER_DISARM = 0x00004102

    def __init__(self, dev_path: str = TIMER_DEV):
        self._dev_path = os.environ.get("IATO_EL2_TIMER_DEV", dev_path)
        self._fd = os.open(self._dev_path, os.O_RDONLY)
        self._owner = threading.current_thread().ident

    def arm(self, interval_ns: int) -> None:
        rc = fcntl.ioctl(self._fd, self.IOCTL_TIMER_ARM, int(interval_ns).to_bytes(8, "little"))
        get_hw_journal().record("cnthp", "ioctl_arm", data={"interval_ns": interval_ns, "rc": int(0 if rc is None else rc)})
        get_hw_journal().record("cnthp", "timer_armed", data={"interval_ns": interval_ns, "backend": "hw"})

    def disarm(self) -> None:
        rc = fcntl.ioctl(self._fd, self.IOCTL_TIMER_DISARM)
        get_hw_journal().record("cnthp", "ioctl_disarm", data={"rc": int(0 if rc is None else rc)})
        get_hw_journal().record("cnthp", "timer_disarmed", data={})

    def wait_for_expiry(self) -> None:
        if threading.current_thread().ident != self._owner:
            raise RuntimeError("wait_for_expiry must be called from owning thread")
        start = time.monotonic_ns()
        os.read(self._fd, 8)
        get_hw_journal().record("cnthp", "timer_fired", data={"elapsed_ns": time.monotonic_ns() - start, "backend": "hw"})

    def close(self) -> None:
        os.close(self._fd)
