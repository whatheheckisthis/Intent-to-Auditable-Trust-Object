from __future__ import annotations

import hashlib
import os
from typing import Callable

from src.spdm_transport import SpdmLoopbackTransport, SpdmTransport


class SpdmError(RuntimeError):
    pass


class SpdmStateMachine:
    def __init__(self, ca_cert_der: bytes, transport: SpdmTransport | None = None, rng: Callable[[int], bytes] | None = None) -> None:
        self.ca_cert_der = ca_cert_der
        self._transport = transport or SpdmLoopbackTransport()
        self._rng = rng or os.urandom
        self.state = "INIT"
        self.session_id = b""
        self._transcript = bytearray()
        self.transcript_hash = b""

    def _step(self, send_msg: bytes) -> None:
        self._transport.send(send_msg)
        recv = self._transport.recv(4096)
        self._transcript.extend(send_msg + recv)

    def run_full_handshake(self) -> None:
        for msg, st in [(b"GET_VERSION", "VERSION"), (b"GET_CAPABILITIES", "CAPS"), (b"NEGOTIATE_ALGORITHMS", "ALGS"), (b"CHALLENGE", "ATTESTED")]:
            self._step(msg)
            self.state = st
        self.session_id = self._rng(16)
        self.transcript_hash = hashlib.sha256(bytes(self._transcript)).digest()
        self._transcript.clear()
