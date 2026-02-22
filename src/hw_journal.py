from __future__ import annotations

import json
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EVENT_CATALOGUE: dict[str, set[str]] = {
    "tpm": {"pcr_read", "pcr_extend", "seal_called", "verify_called", "hw_connect"},
    "spdm": {
        "state_transition",
        "message_sent",
        "message_received",
        "transcript_hash",
        "session_id",
        "cert_verified",
        "transport_connect",
    },
    "smmu": {"ste_write", "ste_fault", "ste_revoke", "cmdq_sync", "state_transition", "mmio_open"},
    "cnthp": {"timer_armed", "timer_fired", "timer_disarmed", "sweep_start", "sweep_complete", "ioctl_arm", "ioctl_disarm"},
    "validator": {"validate_start", "length_check", "expiry_check", "nonce_check", "spdm_binding", "ecdsa_verify", "validate_complete"},
    "orchestrator": {"provision_start", "provision_complete", "shutdown_start", "shutdown_complete"},
}


class HwJournal:
    def __init__(self, journal_dir: Path = Path("build/hw-journal")):
        self.journal_dir = journal_dir
        self.journal_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S.%fZ")
        self.path = self.journal_dir / f"{ts}.ndjson"
        self._fh = self.path.open("a", encoding="utf-8")
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=4096)
        self._entries: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._writer_lock = threading.Lock()
        self._closed = False
        self._thread = threading.Thread(target=self._writer, daemon=True)
        self._thread.start()
        self.dropped = 0
        self.total_events = 0
        self.concern_counts: dict[str, int] = {}
        self._current_test: str | None = None
        self._snapshot_start = 0

    def attach_test(self, nodeid: str) -> None:
        with self._lock:
            self._current_test = nodeid
            self._snapshot_start = len(self._entries)

    def detach_test(self) -> None:
        with self._lock:
            self._current_test = None

    def snapshot(self) -> list[dict[str, Any]]:
        try:
            self._queue.join()
        except Exception:
            pass
        with self._lock:
            return [dict(e) for e in self._entries[self._snapshot_start :]]

    def record(self, concern: str, event: str, stream_id: int | None = None, data: dict | None = None) -> None:
        try:
            if concern not in EVENT_CATALOGUE or event not in EVENT_CATALOGUE[concern]:
                return
            with self._lock:
                test = self._current_test
            entry = {
                "ts": time.time(),
                "monotonic": time.monotonic(),
                "concern": concern,
                "event": event,
                "stream_id": stream_id,
                "data": data or {},
                "test": test,
            }
            self._queue.put_nowait(entry)
        except Exception:
            self.dropped += 1

    def _writer(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                return
            try:
                line = json.dumps(item, sort_keys=True)
                with self._writer_lock:
                    self._fh.write(f"{line}\n")
                    self._fh.flush()
                with self._lock:
                    self._entries.append(item)
                    self.total_events += 1
                    self.concern_counts[item["concern"]] = self.concern_counts.get(item["concern"], 0) + 1
            except Exception:
                self.dropped += 1
            finally:
                self._queue.task_done()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._queue.put_nowait(None)
        except Exception:
            self.dropped += 1
        try:
            self._thread.join(timeout=2)
        except Exception:
            pass
        try:
            with self._writer_lock:
                self._fh.close()
        except Exception:
            pass


_JOURNAL: HwJournal | None = None


def get_hw_journal() -> HwJournal:
    global _JOURNAL
    if _JOURNAL is None:
        _JOURNAL = HwJournal()
    return _JOURNAL
