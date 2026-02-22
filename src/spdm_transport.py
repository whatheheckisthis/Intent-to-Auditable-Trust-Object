from __future__ import annotations

import socket
import struct
import time
from collections import deque
from typing import Protocol

from src.hw_journal import get_hw_journal


class SpdmTransportError(RuntimeError):
    pass


class SpdmTransport(Protocol):
    def send(self, data: bytes) -> None: ...
    def recv(self, max_bytes: int) -> bytes: ...
    def close(self) -> None: ...


class SpdmQemuTransport:
    DEFAULT_SOCKET = "/tmp/iato-spdm.sock"
    CONNECT_TIMEOUT_S = 5.0
    RECV_TIMEOUT_S = 10.0

    def __init__(self, socket_path: str = DEFAULT_SOCKET):
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.settimeout(self.CONNECT_TIMEOUT_S)
        try:
            self._sock.connect(socket_path)
            get_hw_journal().record("spdm", "transport_connect", data={"socket_path": socket_path, "success": True, "error": None})
        except OSError as exc:
            get_hw_journal().record("spdm", "transport_connect", data={"socket_path": socket_path, "success": False, "error": str(exc)})
            self._sock.close()
            raise SpdmTransportError("SPDM connect timeout/failure") from exc
        self._sock.settimeout(self.RECV_TIMEOUT_S)

    def send(self, data: bytes) -> None:
        self._sock.sendall(struct.pack(">I", len(data)) + data)

    def recv(self, max_bytes: int) -> bytes:
        hdr = self._sock.recv(4)
        if len(hdr) < 4:
            raise SpdmTransportError("Incomplete frame header")
        length = struct.unpack(">I", hdr)[0]
        payload = self._sock.recv(length)
        return payload[:max_bytes]

    def close(self) -> None:
        if self._sock:
            self._sock.close()

    def __enter__(self) -> "SpdmQemuTransport":
        return self

    def __exit__(self, *_) -> None:
        self.close()


class SpdmLoopbackTransport:
    def __init__(self):
        self._q: deque[bytes] = deque()
        self._closed = False

    def send(self, data: bytes) -> None:
        if self._closed:
            raise SpdmTransportError("transport closed")
        self._q.append(data)

    def recv(self, max_bytes: int) -> bytes:
        if not self._q:
            raise SpdmTransportError("no data available")
        return self._q.popleft()[:max_bytes]

    def close(self) -> None:
        self._closed = True
