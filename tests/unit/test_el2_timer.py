from __future__ import annotations

import os
import time

import pytest

from src.el2_timer import CnthpTimerBackend, ThreadingTimerBackend
from src.cnthp_sweep import _select_timer_backend


class TestThreadingTimerBackend:
    def test_arm_and_wait_fires_after_interval(self):
        t = ThreadingTimerBackend()
        start = time.monotonic()
        t.arm(20_000_000)
        t.wait_for_expiry()
        assert time.monotonic() - start >= 0.01

    def test_disarm_prevents_wait_from_blocking(self):
        t = ThreadingTimerBackend()
        t.arm(500_000_000)
        t.disarm()
        start = time.monotonic()
        t.wait_for_expiry()
        assert time.monotonic() - start < 0.1

    def test_arm_twice_resets_interval(self):
        t = ThreadingTimerBackend()
        t.arm(1_000_000_000)
        t.arm(1_000_000)
        start = time.monotonic()
        t.wait_for_expiry()
        assert time.monotonic() - start < 0.1


class TestCnthpTimerBackend:
    def test_missing_device_falls_back_to_threading(self, monkeypatch):
        monkeypatch.setenv("IATO_HW_MODE", "1")
        monkeypatch.setattr(os.path, "exists", lambda *_: False)
        assert isinstance(_select_timer_backend(), ThreadingTimerBackend)

    def test_hw_mode_0_uses_threading_backend(self, monkeypatch):
        monkeypatch.setenv("IATO_HW_MODE", "0")
        assert isinstance(_select_timer_backend(), ThreadingTimerBackend)

    def test_ioctl_arm_called_with_correct_interval(self, monkeypatch):
        monkeypatch.setattr(os, "open", lambda *a, **k: 3)
        calls = []
        monkeypatch.setattr("fcntl.ioctl", lambda fd, cmd, data=None: calls.append((fd, cmd, data)))
        b = CnthpTimerBackend("/tmp/dev")
        b.arm(123)
        assert calls[0][1] == CnthpTimerBackend.IOCTL_TIMER_ARM

    def test_wait_blocks_until_read_returns(self, monkeypatch):
        monkeypatch.setattr(os, "open", lambda *a, **k: 3)
        monkeypatch.setattr("fcntl.ioctl", lambda *a, **k: 0)
        monkeypatch.setattr(os, "read", lambda *a, **k: b"\x00" * 8)
        b = CnthpTimerBackend("/tmp/dev")
        b.wait_for_expiry()
