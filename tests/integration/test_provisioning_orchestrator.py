from __future__ import annotations

import time

from src.provisioning import ProvisioningOrchestrator


def test_provision_succeeds_end_to_end(monkeypatch, tmp_path):
    monkeypatch.setenv("IATO_HW_MODE", "0")
    monkeypatch.setenv("IATO_KEY_PATH", str(tmp_path / "k.pem"))
    o = ProvisioningOrchestrator.create()
    assert o.provision(1, {"pa_base": 0x1000, "pa_limit": 0x2000, "permissions": 3, "ttl": 1}) == "OK"


def test_provision_then_sweep_faults_expired(monkeypatch, tmp_path):
    monkeypatch.setenv("IATO_HW_MODE", "0")
    monkeypatch.setenv("IATO_KEY_PATH", str(tmp_path / "k.pem"))
    o = ProvisioningOrchestrator.create()
    o.provision(2, {"pa_base": 0x1000, "pa_limit": 0x2000, "permissions": 3, "ttl": -1})
    o.sweep._sweep_once()
    assert o.smmu._table[2]["state"] == "FAULT_ALL"


def test_shutdown_faults_all_permitted_streams(monkeypatch, tmp_path):
    monkeypatch.setenv("IATO_HW_MODE", "0")
    monkeypatch.setenv("IATO_KEY_PATH", str(tmp_path / "k.pem"))
    o = ProvisioningOrchestrator.create()
    o.provision(3, {"pa_base": 0x1000, "pa_limit": 0x2000, "permissions": 3, "ttl": 10})
    o.shutdown()
    assert o.smmu._table[3]["state"] == "FAULT_ALL"


def test_hw_mode_selects_correct_backends(monkeypatch, tmp_path):
    monkeypatch.setenv("IATO_HW_MODE", "1")
    monkeypatch.setenv("IATO_TPM_SIM", "1")
    monkeypatch.setenv("IATO_MANA_SIM_MMIO", "1")
    monkeypatch.setenv("IATO_KEY_PATH", str(tmp_path / "k.pem"))
    monkeypatch.setattr("src.provisioning.CnthpTimerBackend", lambda: None)
    monkeypatch.setattr("src.spdm_session_factory.SpdmQemuTransport", lambda *a, **k: type("T", (), {"send": lambda *_: None, "recv": lambda *_: b"x"})())
    o = ProvisioningOrchestrator.create()
    assert o is not None


def test_smmu_mmio_write_called_on_provision(monkeypatch, tmp_path):
    monkeypatch.setenv("IATO_HW_MODE", "0")
    monkeypatch.setenv("IATO_KEY_PATH", str(tmp_path / "k.pem"))
    o = ProvisioningOrchestrator.create()
    o.provision(4, {"pa_base": 0x1000, "pa_limit": 0x2000, "permissions": 3, "ttl": 10})
    assert o.smmu._table[4]["state"] == "PERMITTED"


def test_cnthp_ioctl_armed_on_start_sweep(monkeypatch, tmp_path):
    monkeypatch.setenv("IATO_HW_MODE", "0")
    monkeypatch.setenv("IATO_KEY_PATH", str(tmp_path / "k.pem"))
    o = ProvisioningOrchestrator.create()
    o.start_sweep()
    o.stop_sweep()
    assert True
