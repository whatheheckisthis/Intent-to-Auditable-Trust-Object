from __future__ import annotations

import fcntl
import os
import threading
import time
from typing import Protocol


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

    def disarm(self) -> None:
        self._event.set()

    def wait_for_expiry(self) -> None:
        timeout = max(0, self._deadline - time.monotonic())
        self._event.wait(timeout)


class CnthpTimerBackend:
    TIMER_DEV = "/dev/iato-el2-timer"
    IOCTL_TIMER_ARM = 0xC0084101
    IOCTL_TIMER_DISARM = 0x00004102

    def __init__(self, dev_path: str = TIMER_DEV):
        self._dev_path = os.environ.get("IATO_EL2_TIMER_DEV", dev_path)
        self._fd = os.open(self._dev_path, os.O_RDONLY)
        self._owner = threading.current_thread().ident

    def arm(self, interval_ns: int) -> None:
        fcntl.ioctl(self._fd, self.IOCTL_TIMER_ARM, int(interval_ns).to_bytes(8, "little"))

    def disarm(self) -> None:
        fcntl.ioctl(self._fd, self.IOCTL_TIMER_DISARM)

    def wait_for_expiry(self) -> None:
        if threading.current_thread().ident != self._owner:
            raise RuntimeError("wait_for_expiry must be called from owning thread")
        os.read(self._fd, 8)

    def close(self) -> None:
        os.close(self._fd)
