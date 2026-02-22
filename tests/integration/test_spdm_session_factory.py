from __future__ import annotations

from src.spdm_session_factory import make_spdm_session


def test_loopback_session_reaches_attested(monkeypatch):
    monkeypatch.setenv("IATO_HW_MODE", "0")
    sm = make_spdm_session(b"ca")
    assert sm.state == "ATTESTED"


def test_session_id_is_16_bytes(monkeypatch):
    monkeypatch.setenv("IATO_HW_MODE", "0")
    assert len(make_spdm_session(b"ca").session_id) == 16


def test_transcript_hash_not_empty(monkeypatch):
    monkeypatch.setenv("IATO_HW_MODE", "0")
    assert make_spdm_session(b"ca").transcript_hash


def test_hw_mode_uses_qemu_transport(monkeypatch):
    monkeypatch.setenv("IATO_HW_MODE", "1")
    monkeypatch.setattr("src.spdm_session_factory.SpdmQemuTransport", lambda *a, **k: type("T", (), {"send": lambda *_: None, "recv": lambda *_: b"x"})())
    sm = make_spdm_session(b"ca")
    assert sm.state == "ATTESTED"
