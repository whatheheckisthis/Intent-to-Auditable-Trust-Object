import pytest

from iato_ops.api_keys import APIKeyManager


def test_get_policy_service_key_reads_env(monkeypatch):
    monkeypatch.setenv("IATO_POLICY_SERVICE_API_KEY", "policy-secret")
    assert APIKeyManager().get_policy_service_key() == "policy-secret"


def test_get_telemetry_service_key_reads_env(monkeypatch):
    monkeypatch.setenv("IATO_TELEMETRY_API_KEY", "telemetry-secret")
    assert APIKeyManager().get_telemetry_service_key() == "telemetry-secret"


def test_missing_key_raises_value_error(monkeypatch):
    monkeypatch.delenv("IATO_POLICY_SERVICE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="IATO_POLICY_SERVICE_API_KEY"):
        APIKeyManager().get_policy_service_key()
