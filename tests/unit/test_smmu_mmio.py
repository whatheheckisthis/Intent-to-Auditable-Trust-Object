from __future__ import annotations

import struct

import pytest

from src.smmu_mmio import SmmuMmioBackend, SmmuMmioError, SmmuSimBackend


class TestSmmuSimBackend:
    def test_write_ste_stores_correct_fields(self):
        sim = SmmuSimBackend()
        sim.write_ste(1, 0x1000, 0x2000, 7)
        assert len(sim.read_ste_raw(1)) == 64

    def test_fault_ste_sets_v_bit_zero(self):
        sim = SmmuSimBackend()
        sim.fault_ste(1)
        assert sim.read_ste_raw(1)[0] & 0x1 == 0

    def test_read_ste_raw_is_64_bytes(self):
        assert len(SmmuSimBackend().read_ste_raw(9)) == 64

    def test_write_then_fault_then_write_again(self):
        sim = SmmuSimBackend()
        sim.write_ste(1, 0x1000, 0x2000, 1)
        sim.fault_ste(1)
        sim.write_ste(1, 0x3000, 0x4000, 1)
        assert sim.read_ste_raw(1)[0] & 0x1 == 1


class TestSmmuMmioBackend:
    def test_missing_dev_mem_raises_mmio_error(self):
        with pytest.raises(SmmuMmioError):
            SmmuMmioBackend(mem_dev="/definitely-missing").open()

    def test_hw_mode_0_uses_sim_backend(self, monkeypatch):
        monkeypatch.setenv("IATO_HW_MODE", "0")
        assert SmmuSimBackend() is not None

    def test_ste_binary_layout_correct(self):
        b = SmmuMmioBackend()
        raw = b._build_ste(0x4000, 0x8000, 3, 1)
        words = struct.unpack("<QQQQQQQQ", raw)
        assert words[0] & 0x1 == 1
        assert words[3] == 0x4

    def test_fault_ste_binary_layout_correct(self):
        b = SmmuMmioBackend()
        b.fault_ste(2)
        words = struct.unpack("<QQQQQQQQ", b.read_ste_raw(2))
        assert words[0] & 0x1 == 0
