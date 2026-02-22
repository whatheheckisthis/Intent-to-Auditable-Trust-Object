from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

from src.hw_journal import get_hw_journal


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
            self._hw_connect()

    def _hw_connect(self) -> None:
        journal = get_hw_journal()
        try:
            from tpm2_pytss import ESAPI  # type: ignore

            self._esys = ESAPI()
            journal.record("tpm", "hw_connect", data={"dev_path": "esapi", "success": True, "error": None})
        except Exception as exc:  # pragma: no cover - hardware path
            journal.record("tpm", "hw_connect", data={"dev_path": "esapi", "success": False, "error": str(exc)})
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
        journal = get_hw_journal()
        journal.record("tpm", "seal_called", data={"trust_store_sha256": hashlib.sha256(measurement).hexdigest()})
        if self._hw_mode:
            value = self._hw_pcr_extend(measurement)
            journal.record("tpm", "pcr_extend", data={"pcr_index": self.pcr_index, "measurement_hex": measurement.hex(), "new_value_hex": value})
            return value
        self._sim_pcr = self._simulate_pcr_extend(self._sim_pcr, measurement)
        journal.record("tpm", "pcr_extend", data={"pcr_index": self.pcr_index, "measurement_hex": measurement.hex(), "new_value_hex": self._sim_pcr})
        return self._sim_pcr

    def read_pcr(self) -> str:
        value = self._hw_pcr_read() if self._hw_mode else self._sim_pcr
        get_hw_journal().record("tpm", "pcr_read", data={"pcr_index": self.pcr_index, "value_hex": value})
        return value

    def verify_enrollment_unchanged(self) -> bool:
        if self._sealed_measurement is None:
            return False
        expected = self._simulate_pcr_extend("00" * 32, self._sealed_measurement)
        actual = self.read_pcr()
        result = actual == expected
        get_hw_journal().record("tpm", "verify_called", data={"result": result, "expected_hex": expected, "actual_hex": actual})
        return result

    def reset_simulation(self) -> None:
        self._sim_pcr = "00" * 32
        self._sealed_measurement = None
