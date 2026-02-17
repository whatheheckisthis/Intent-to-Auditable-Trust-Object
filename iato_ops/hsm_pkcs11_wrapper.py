"""PKCS#11 wrapper for HSM-backed key lifecycle operations."""

from __future__ import annotations

import base64
import hashlib
import importlib
import os
import secrets
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PublicKeyMaterial:
    """Public key metadata returned from HSM keypair generation."""

    label: str
    pem: str
    fingerprint: str


class PKCS11Wrapper:
    """Thin adapter around ``python-pkcs11``.

    If a PKCS#11 provider is unavailable, a deterministic simulator is used so the
    RBAC/audit/provisioning flow can be exercised locally.
    """

    def __init__(self, expected_user_pin: str, pkcs11_module_path: str | None = None) -> None:
        self.expected_user_pin = expected_user_pin
        self.pkcs11_module_path = pkcs11_module_path or os.getenv("PKCS11_MODULE")
        self.session: Any | None = None
        self.session_id: str | None = None
        self._simulated = False
        self._sim_private_keys: dict[str, bytes] = {}

        pkcs11_module = importlib.import_module("pkcs11") if importlib.util.find_spec("pkcs11") else None
        self._pkcs11 = pkcs11_module
        self._lib = pkcs11_module.lib(self.pkcs11_module_path) if pkcs11_module and self.pkcs11_module_path else None

    def verify_user_pin(self, supplied_pin: str) -> bool:
        return secrets.compare_digest(self.expected_user_pin, supplied_pin)

    def open_session(self, token_label: str) -> str:
        """Open an authenticated session and return session id."""

        if self._lib is None:
            self._simulated = True
            self.session_id = f"sim-{secrets.token_hex(8)}"
            return self.session_id

        token = self._lib.get_token(token_label=token_label)
        self.session = token.open(user_pin=self.expected_user_pin, rw=True)
        self.session_id = str(id(self.session))
        return self.session_id

    def close_session(self) -> None:
        if self.session is not None:
            self.session.close()
            self.session = None

    def generate_keypair(self, label: str) -> PublicKeyMaterial:
        """Generate a non-extractable keypair in HSM and return public material."""

        if self._simulated:
            private_seed = secrets.token_bytes(32)
            public_blob = hashlib.sha256(private_seed + label.encode()).digest()
            self._sim_private_keys[label] = private_seed
            return PublicKeyMaterial(
                label=label,
                pem=self._der_to_pem(public_blob),
                fingerprint=hashlib.sha256(public_blob).hexdigest(),
            )

        if self.session is None:
            raise RuntimeError("HSM session not opened.")

        key_type = self._pkcs11.KeyType.RSA
        mechanism = self._pkcs11.Mechanism.RSA_PKCS_KEY_PAIR_GEN
        pub, _priv = self.session.generate_keypair(
            key_type,
            3072,
            store=True,
            label=label,
            public_template={self._pkcs11.Attribute.ENCRYPT: True, self._pkcs11.Attribute.VERIFY: True},
            private_template={
                self._pkcs11.Attribute.SIGN: True,
                self._pkcs11.Attribute.DECRYPT: True,
                self._pkcs11.Attribute.EXTRACTABLE: False,
                self._pkcs11.Attribute.SENSITIVE: True,
            },
            mechanism=mechanism,
        )
        public_der = bytes(pub[self._pkcs11.Attribute.VALUE])
        return PublicKeyMaterial(
            label=label,
            pem=self._der_to_pem(public_der),
            fingerprint=hashlib.sha256(public_der).hexdigest(),
        )

    def sign(self, key_label: str, payload: bytes) -> bytes:
        """Sign payload using HSM private key handle."""

        if self._simulated:
            private_seed = self._sim_private_keys[key_label]
            return hashlib.sha256(private_seed + payload).digest()

        if self.session is None:
            raise RuntimeError("HSM session not opened.")

        priv = self.session.get_key(label=key_label, key_type=self._pkcs11.KeyType.RSA, object_class=self._pkcs11.ObjectClass.PRIVATE_KEY)
        return bytes(priv.sign(payload, mechanism=self._pkcs11.Mechanism.SHA256_RSA_PKCS))

    def encrypt(self, key_label: str, plaintext: bytes) -> bytes:
        """Encrypt payload using HSM-managed public key handle."""

        if self._simulated:
            return base64.b64encode(plaintext)

        if self.session is None:
            raise RuntimeError("HSM session not opened.")

        pub = self.session.get_key(label=key_label, key_type=self._pkcs11.KeyType.RSA, object_class=self._pkcs11.ObjectClass.PUBLIC_KEY)
        return bytes(pub.encrypt(plaintext, mechanism=self._pkcs11.Mechanism.RSA_PKCS))

    @staticmethod
    def _der_to_pem(der_bytes: bytes) -> str:
        b64 = base64.encodebytes(der_bytes).decode().replace("\n", "")
        lines = [b64[i : i + 64] for i in range(0, len(b64), 64)]
        return "-----BEGIN PUBLIC KEY-----\n" + "\n".join(lines) + "\n-----END PUBLIC KEY-----"
