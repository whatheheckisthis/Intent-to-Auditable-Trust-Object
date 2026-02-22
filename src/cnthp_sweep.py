from __future__ import annotations

import os
import threading
import time
from typing import Callable

from src.el2_timer import CnthpTimerBackend, El2TimerBackend, ThreadingTimerBackend
from src.hw_journal import get_hw_journal


def _select_timer_backend() -> El2TimerBackend:
    if os.environ.get("IATO_HW_MODE", "0") == "1" and os.path.exists(os.environ.get("IATO_EL2_TIMER_DEV", CnthpTimerBackend.TIMER_DEV)):
        try:
            return CnthpTimerBackend()
        except OSError:
            pass
    return ThreadingTimerBackend()


class CnthpSweep:
    def __init__(self, smmu, binding_table: dict, interval_sec: int = 30, clock: Callable[[], float] | None = None, timer: El2TimerBackend | None = None) -> None:
        self._smmu = smmu
        self._binding_table = binding_table
        self._interval = interval_sec
        self._clock = clock or time.time
        self._timer = timer or _select_timer_backend()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _sweep_once(self) -> None:
        start = time.monotonic_ns()
        now = self._clock()
        journal = get_hw_journal()
        journal.record("cnthp", "sweep_start", data={"active_streams": len(self._binding_table)})
        swept: list[int] = []
        for sid, rec in list(self._binding_table.items()):
            if rec.get("expires_at", now + 1) <= now:
                self._smmu.fault_everything(sid)
                swept.append(sid)
        journal.record("cnthp", "sweep_complete", data={"swept": swept, "duration_ns": time.monotonic_ns() - start})

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._timer.arm(self._interval * 1_000_000_000)
            self._timer.wait_for_expiry()
            if self._stop_event.is_set():
                break
            self._sweep_once()
        self._timer.disarm()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._timer.disarm()
        if self._thread:
            self._thread.join(timeout=1)
