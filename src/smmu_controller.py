from __future__ import annotations

import os
from dataclasses import dataclass

from src.smmu_mmio import SmmuMmioBackend, SmmuMmioError, SmmuSimBackend


class SmmuError(RuntimeError):
    pass


@dataclass
class SteCredential:
    stream_id: int
    pa_range_base: int
    pa_range_limit: int
    permissions: int


class SmmuController:
    def __init__(self, mmio_base: int | None = None, mmio: SmmuMmioBackend | SmmuSimBackend | None = None) -> None:
        self._table: dict[int, dict] = {}
        if mmio is not None:
            self._mmio = mmio
        elif mmio_base is not None and os.environ.get("IATO_HW_MODE") == "1" and os.environ.get("IATO_MANA_SIM_MMIO", "0") != "1":
            self._mmio = SmmuMmioBackend(mmio_base)
            self._mmio.open()
        else:
            self._mmio = SmmuSimBackend()

    def write_ste(self, credential: SteCredential) -> None:
        self._table[credential.stream_id] = {"state": "PERMITTED", "credential": credential}
        try:
            self._mmio.write_ste(credential.stream_id, credential.pa_range_base, credential.pa_range_limit, credential.permissions)
        except SmmuMmioError as exc:
            self._table[credential.stream_id]["state"] = "FAULT_ALL"
            raise SmmuError(str(exc)) from exc

    def fault_everything(self, stream_id: int) -> None:
        self._table[stream_id] = {"state": "FAULT_ALL"}
        self._mmio.fault_ste(stream_id)
