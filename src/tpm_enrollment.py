from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass


class TpmEnrollmentError(RuntimeError):
    pass


@dataclass
class TpmEnrollmentSeal:
    pcr_index: int = 16

    def __post_init__(self) -> None:
        self._sim_pcr = "00" * 32
        self._sealed_measurement: bytes | None = None
        self._esys = None
        self._hw_mode = os.environ.get("IATO_HW_MODE", "0") == "1" and os.environ.get("IATO_TPM_SIM", "0") != "1"
        if self._hw_mode:
            try:
                from tpm2_pytss import ESAPI  # type: ignore

                self._esys = ESAPI()
            except Exception as exc:  # pragma: no cover - hardware path
                raise TpmEnrollmentError("Unable to initialize TPM ESAPI") from exc

    @staticmethod
    def _simulate_pcr_extend(current_hex: str, measurement: bytes) -> str:
        current = bytes.fromhex(current_hex)
        return hashlib.sha256(current + hashlib.sha256(measurement).digest()).hexdigest()

    def _hw_pcr_extend(self, measurement: bytes) -> str:
        raise TpmEnrollmentError("Hardware TPM extend not available in this environment")

    def _hw_pcr_read(self) -> str:
        raise TpmEnrollmentError("Hardware TPM read not available in this environment")

    def seal(self, measurement: bytes) -> str:
        self._sealed_measurement = measurement
        if self._hw_mode:
            return self._hw_pcr_extend(measurement)
        self._sim_pcr = self._simulate_pcr_extend(self._sim_pcr, measurement)
        return self._sim_pcr

    def read_pcr(self) -> str:
        return self._hw_pcr_read() if self._hw_mode else self._sim_pcr

    def verify_enrollment_unchanged(self) -> bool:
        if self._sealed_measurement is None:
            return False
        expected = self._simulate_pcr_extend("00" * 32, self._sealed_measurement)
        return self.read_pcr() == expected

    def reset_simulation(self) -> None:
        self._sim_pcr = "00" * 32
        self._sealed_measurement = None
