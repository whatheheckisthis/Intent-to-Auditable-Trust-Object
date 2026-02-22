from __future__ import annotations

import mmap
import os
import struct

SMMU_IDR0 = 0x0000
SMMU_IDR1 = 0x0004
SMMU_CR0 = 0x0020
SMMU_CR0ACK = 0x0024
SMMU_STRTAB_BASE = 0x0080
SMMU_STRTAB_BASE_CFG = 0x0088
SMMU_CMDQ_BASE = 0x0090
SMMU_CMDQ_PROD = 0x009C
SMMU_CMDQ_CONS = 0x00A0
STE_SIZE_BYTES = 64


class SmmuMmioError(RuntimeError):
    pass


class SmmuMmioBackend:
    DEFAULT_MMIO_BASE = 0x09050000
    STRTAB_SIZE = 4096 * 64
    CMDQ_TIMEOUT_ITERS = 1000

    def __init__(self, mmio_base: int = DEFAULT_MMIO_BASE, mem_dev: str = "/dev/mem"):
        self.mmio_base = mmio_base
        self.mem_dev = mem_dev
        self._fd = None
        self._map = None
        self._raw: dict[int, bytes] = {}

    def open(self) -> None:
        try:
            self._fd = os.open(self.mem_dev, os.O_RDWR | os.O_SYNC)
            self._map = mmap.mmap(self._fd, self.STRTAB_SIZE, mmap.MAP_SHARED, mmap.PROT_WRITE | mmap.PROT_READ, offset=self.mmio_base)
        except OSError as exc:
            raise SmmuMmioError("mmap failed") from exc

    def close(self) -> None:
        if self._map is not None:
            self._map.close()
            self._map = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def _build_ste(self, pa_base: int, pa_limit: int, permissions: int, valid: int) -> bytes:
        word0 = (valid & 0x1) | (0b100 << 1 if valid else 0)
        word3 = pa_base >> 12
        return struct.pack("<QQQQQQQQ", word0, permissions, pa_limit - pa_base, word3, 0, 0, 0, 0)

    def write_ste(self, stream_id: int, pa_base: int, pa_limit: int, permissions: int) -> None:
        ste = self._build_ste(pa_base, pa_limit, permissions, 1)
        self._raw[stream_id] = ste
        if self._map is not None:
            off = stream_id * STE_SIZE_BYTES
            self._map[off: off + STE_SIZE_BYTES] = ste

    def fault_ste(self, stream_id: int) -> None:
        ste = self._build_ste(0, 0x1000, 0, 0)
        self._raw[stream_id] = ste
        if self._map is not None:
            off = stream_id * STE_SIZE_BYTES
            self._map[off: off + STE_SIZE_BYTES] = ste

    def read_ste_raw(self, stream_id: int) -> bytes:
        return self._raw.get(stream_id, b"\x00" * 64)

    def __enter__(self) -> "SmmuMmioBackend":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()


class SmmuSimBackend:
    def __init__(self):
        self._table = {}

    def write_ste(self, stream_id, pa_base, pa_limit, permissions) -> None:
        self._table[stream_id] = {"pa_base": pa_base, "pa_limit": pa_limit, "permissions": permissions, "valid": 1}

    def fault_ste(self, stream_id: int) -> None:
        self._table[stream_id] = {"pa_base": 0, "pa_limit": 0, "permissions": 0, "valid": 0}

    def read_ste_raw(self, stream_id: int) -> bytes:
        e = self._table.get(stream_id, {"pa_base": 0, "pa_limit": 0, "permissions": 0, "valid": 0})
        word0 = (e["valid"] & 0x1) | (0b100 << 1 if e["valid"] else 0)
        return struct.pack("<QQQQQQQQ", word0, e["permissions"], e["pa_limit"] - e["pa_base"], e["pa_base"] >> 12, 0, 0, 0, 0)
