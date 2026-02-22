from __future__ import annotations

import json
import threading
from pathlib import Path

from src.hw_journal import HwJournal


class TestHwJournal:
    def test_record_writes_valid_ndjson(self, tmp_path: Path):
        j = HwJournal(tmp_path)
        j.record("tpm", "pcr_read", data={"pcr_index": 16, "value_hex": "00" * 32})
        j.close()
        lines = j.path.read_text(encoding="utf-8").strip().splitlines()
        assert lines
        json.loads(lines[0])

    def test_record_never_raises_on_write_failure(self, tmp_path: Path, monkeypatch):
        j = HwJournal(tmp_path)
        monkeypatch.setattr(j._queue, "put_nowait", lambda *_: (_ for _ in ()).throw(OSError("boom")))
        j.record("tpm", "pcr_read", data={"pcr_index": 16, "value_hex": "00" * 32})
        j.close()

    def test_attach_detach_test_sets_context(self, tmp_path: Path):
        j = HwJournal(tmp_path)
        j.attach_test("a::b")
        j.record("tpm", "pcr_read", data={"pcr_index": 16, "value_hex": "00" * 32})
        snap = j.snapshot()
        assert snap[0]["test"] == "a::b"
        j.detach_test()
        j.close()

    def test_snapshot_returns_only_current_test_events(self, tmp_path: Path):
        j = HwJournal(tmp_path)
        j.attach_test("t1")
        j.record("tpm", "pcr_read", data={"pcr_index": 16, "value_hex": "00" * 32})
        _ = j.snapshot()
        j.detach_test()
        j.attach_test("t2")
        j.record("tpm", "pcr_read", data={"pcr_index": 16, "value_hex": "11" * 32})
        snap = j.snapshot()
        assert len(snap) == 1
        assert snap[0]["test"] == "t2"
        j.close()

    def test_dropped_count_increments_on_failure(self, tmp_path: Path, monkeypatch):
        j = HwJournal(tmp_path)
        monkeypatch.setattr(j._queue, "put_nowait", lambda *_: (_ for _ in ()).throw(RuntimeError("full")))
        j.record("tpm", "pcr_read", data={"pcr_index": 16, "value_hex": "00" * 32})
        assert j.dropped >= 1
        j.close()

    def test_entries_are_monotonically_ordered(self, tmp_path: Path):
        j = HwJournal(tmp_path)
        j.record("tpm", "pcr_read", data={"pcr_index": 16, "value_hex": "00" * 32})
        j.record("tpm", "pcr_read", data={"pcr_index": 16, "value_hex": "11" * 32})
        j.close()
        entries = [json.loads(l) for l in j.path.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert entries[1]["monotonic"] >= entries[0]["monotonic"]

    def test_thread_safe_concurrent_writes(self, tmp_path: Path):
        j = HwJournal(tmp_path)

        def worker():
            for _ in range(100):
                j.record("tpm", "pcr_read", data={"pcr_index": 16, "value_hex": "00" * 32})

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        j.close()
        lines = j.path.read_text(encoding="utf-8").splitlines()
        assert len(lines) + j.dropped >= 800

    def test_close_flushes_and_releases_file(self, tmp_path: Path):
        j = HwJournal(tmp_path)
        j.record("tpm", "pcr_read", data={"pcr_index": 16, "value_hex": "00" * 32})
        j.close()
        assert j._fh.closed
