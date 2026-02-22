from __future__ import annotations

import hashlib
import os
from typing import Callable

from src.hw_journal import get_hw_journal
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

    def _send(self, msg: bytes) -> None:
        self._transport.send(msg)
        get_hw_journal().record("spdm", "message_sent", data={"msg_type": msg.decode(errors="replace"), "length_bytes": len(msg)})

    def _recv(self) -> bytes:
        msg = self._transport.recv(4096)
        get_hw_journal().record("spdm", "message_received", data={"msg_type": msg.decode(errors="replace"), "length_bytes": len(msg)})
        return msg

    def _transition(self, to_state: str) -> None:
        prior = self.state
        self.state = to_state
        get_hw_journal().record("spdm", "state_transition", data={"from_state": prior, "to_state": to_state})

    def _compute_transcript_hash(self) -> bytes:
        digest = hashlib.sha256(bytes(self._transcript)).digest()
        get_hw_journal().record("spdm", "transcript_hash", data={"hash_hex": digest.hex()})
        return digest

    def _step(self, send_msg: bytes, to_state: str) -> None:
        self._send(send_msg)
        recv = self._recv()
        self._transcript.extend(send_msg + recv)
        self._transition(to_state)

    def run_full_handshake(self) -> None:
        for msg, st in [(b"GET_VERSION", "VERSION"), (b"GET_CAPABILITIES", "CAPS"), (b"NEGOTIATE_ALGORITHMS", "ALGS"), (b"CHALLENGE", "ATTESTED")]:
            self._step(msg, st)
        self.session_id = self._rng(16)
        get_hw_journal().record("spdm", "session_id", data={"session_id_hex": self.session_id.hex()})
        self.transcript_hash = self._compute_transcript_hash()
        self._transcript.clear()
