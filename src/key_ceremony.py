from __future__ import annotations

import base64
import hashlib
import os
import secrets
from dataclasses import dataclass
from pathlib import Path

from src.tpm_enrollment import TpmEnrollmentSeal


class KeyCeremonyError(RuntimeError):
    pass


@dataclass
class PublicKey:
    der: bytes
    curve_name: str = "secp256r1"


class KeyCeremony:
    SE_ROOT_KEY_PATH = Path("/etc/iato/se_root_key.pem")
    PCR_INDEX = 16

    def __init__(self, tpm: TpmEnrollmentSeal, key_path: Path = SE_ROOT_KEY_PATH):
        self.tpm = tpm
        self.key_path = Path(os.environ.get("IATO_KEY_PATH", str(key_path)))

    def _pub_path(self) -> Path:
        return self.key_path.with_suffix(".pub.pem")

    def _pcr_path(self) -> Path:
        return self.key_path.with_suffix(".pcr")

    def _new_private(self) -> bytes:
        return secrets.token_bytes(32)

    def _public_from_private(self, priv: bytes) -> PublicKey:
        return PublicKey(der=hashlib.sha256(priv).digest() + b"P256")

    def enroll(self) -> PublicKey:
        if self.tpm.read_pcr() != "00" * 32:
            raise KeyCeremonyError("PCR already extended")
        priv = self.key_path.read_bytes() if self.key_path.exists() else self._new_private()
        pub = self._public_from_private(priv)
        trust_store = hashlib.sha256(pub.der).digest() + pub.der
        pcr_val = self.tpm.seal(trust_store)

        self.key_path.parent.mkdir(parents=True, exist_ok=True)
        self.key_path.write_bytes(priv)
        os.chmod(self.key_path, 0o600)
        self._pub_path().write_text(base64.b64encode(pub.der).decode())
        self._pcr_path().write_text(pcr_val)
        return pub

    def re_enroll(self) -> PublicKey:
        if os.environ.get("IATO_HW_MODE", "0") == "1" and os.environ.get("IATO_TPM_SIM", "0") != "1":
            raise KeyCeremonyError("re_enroll blocked in hardware mode")
        self.tpm.reset_simulation()
        return self.enroll()

    def verify(self) -> bool:
        pub_der = base64.b64decode(self._pub_path().read_text(), validate=True)
        trust_store = hashlib.sha256(pub_der).digest() + pub_der
        self.tpm._sealed_measurement = trust_store
        return self.tpm.verify_enrollment_unchanged()

    def load_public_key(self) -> PublicKey:
        if not self.verify():
            raise KeyCeremonyError("PCR verification failed")
        return PublicKey(der=base64.b64decode(self._pub_path().read_text(), validate=True))
