from __future__ import annotations

import socket

import pytest

from src.spdm_transport import SpdmLoopbackTransport, SpdmQemuTransport, SpdmTransportError


class TestLoopbackTransport:
    def test_send_recv_roundtrip(self):
        t = SpdmLoopbackTransport()
        t.send(b"abc")
        assert t.recv(10) == b"abc"

    def test_recv_empty_raises_transport_error(self):
        with pytest.raises(SpdmTransportError):
            SpdmLoopbackTransport().recv(1)

    def test_close_is_idempotent(self):
        t = SpdmLoopbackTransport()
        t.close()
        t.close()


class TestQemuTransport:
    def test_connect_timeout_raises_transport_error(self, monkeypatch):
        class DummySock:
            def settimeout(self, *_): pass
            def connect(self, *_): raise OSError("x")
            def close(self): pass
        monkeypatch.setattr(socket, "socket", lambda *a, **k: DummySock())
        with pytest.raises(SpdmTransportError):
            SpdmQemuTransport("/tmp/missing.sock")

    def test_send_recv_framing_correct(self, monkeypatch):
        sent = bytearray()
        class DummySock:
            def settimeout(self, *_): pass
            def connect(self, *_): pass
            def sendall(self, b): sent.extend(b)
            def recv(self, n): return b"\x00\x00\x00\x03" if n == 4 else b"xyz"
            def close(self): pass
        monkeypatch.setattr(socket, "socket", lambda *a, **k: DummySock())
        t = SpdmQemuTransport("/tmp/x")
        t.send(b"xyz")
        assert sent[:4] == b"\x00\x00\x00\x03"
        assert t.recv(10) == b"xyz"

    def test_context_manager_closes_socket(self, monkeypatch):
        closed = {"v": False}
        class DummySock:
            def settimeout(self, *_): pass
            def connect(self, *_): pass
            def sendall(self, _): pass
            def recv(self, _): return b"\x00\x00\x00\x00"
            def close(self): closed["v"] = True
        monkeypatch.setattr(socket, "socket", lambda *a, **k: DummySock())
        with SpdmQemuTransport("/tmp/x"):
            pass
        assert closed["v"]
