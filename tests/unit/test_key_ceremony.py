from __future__ import annotations

from pathlib import Path

import pytest

from src.key_ceremony import KeyCeremony, KeyCeremonyError
from src.tpm_enrollment import TpmEnrollmentSeal


@pytest.fixture
def tpm_sim():
    return TpmEnrollmentSeal()


class TestEnrollment:
    def test_enroll_produces_p256_public_key(self, tpm_sim, tmp_path):
        pub = KeyCeremony(tpm_sim, tmp_path / "se_root_key.pem").enroll()
        assert pub.curve_name == "secp256r1"

    def test_enroll_extends_pcr_from_zero(self, tpm_sim, tmp_path):
        k = KeyCeremony(tpm_sim, tmp_path / "se_root_key.pem")
        k.enroll()
        assert tpm_sim.read_pcr() != "00" * 32

    def test_enroll_writes_key_files(self, tpm_sim, tmp_path):
        kp = tmp_path / "se_root_key.pem"
        KeyCeremony(tpm_sim, kp).enroll()
        assert kp.exists() and kp.with_suffix(".pub.pem").exists() and kp.with_suffix(".pcr").exists()

    def test_re_enroll_blocked_in_hw_mode(self, monkeypatch, tpm_sim):
        monkeypatch.setenv("IATO_HW_MODE", "1")
        with pytest.raises(KeyCeremonyError):
            KeyCeremony(tpm_sim, Path("/tmp/x.pem")).re_enroll()

    def test_double_enroll_raises_ceremony_error(self, tpm_sim, tmp_path):
        k = KeyCeremony(tpm_sim, tmp_path / "se_root_key.pem")
        k.enroll()
        with pytest.raises(KeyCeremonyError):
            k.enroll()


class TestVerification:
    def test_verify_passes_after_enroll(self, tpm_sim, tmp_path):
        k = KeyCeremony(tpm_sim, tmp_path / "se_root_key.pem")
        k.enroll()
        assert k.verify() is True

    def test_verify_fails_if_key_file_tampered(self, tpm_sim, tmp_path):
        k = KeyCeremony(tpm_sim, tmp_path / "se_root_key.pem")
        k.enroll()
        k._pub_path().write_text("tampered")
        assert k.verify() is False

    def test_load_public_key_calls_verify(self, tpm_sim, tmp_path):
        k = KeyCeremony(tpm_sim, tmp_path / "se_root_key.pem")
        k.enroll()
        assert k.load_public_key() is not None
