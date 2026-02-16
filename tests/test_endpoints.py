import sys
import types

import pytest

# Provide a lightweight requests shim for environments without external deps.
if "requests" not in sys.modules:
    requests_stub = types.ModuleType("requests")
    requests_stub.get = None
    requests_stub.post = None
    sys.modules["requests"] = requests_stub

from iato_ops.endpoints import EndpointClient, EndpointConfig


class DummyResponse:
    def __init__(self, payload, status_ok=True):
        self._payload = payload
        self._status_ok = status_ok

    def raise_for_status(self):
        if not self._status_ok:
            raise RuntimeError("http error")

    def json(self):
        return self._payload


def test_fetch_policy_document_calls_get(monkeypatch):
    captured = {}

    def fake_get(url, headers, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["timeout"] = timeout
        return DummyResponse({"id": "abc"})

    monkeypatch.setattr("iato_ops.endpoints.requests.get", fake_get)

    client = EndpointClient(
        EndpointConfig(policy_base_url="https://policy.local", telemetry_base_url="https://tel.local")
    )
    result = client.fetch_policy_document("abc", "token")

    assert result == {"id": "abc"}
    assert captured["url"] == "https://policy.local/policies/abc"
    assert captured["headers"]["Authorization"] == "Bearer token"
    assert captured["timeout"] == 10


def test_submit_risk_result_calls_post(monkeypatch):
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return DummyResponse({"accepted": True})

    monkeypatch.setattr("iato_ops.endpoints.requests.post", fake_post)

    client = EndpointClient(
        EndpointConfig(policy_base_url="https://policy.local", telemetry_base_url="https://tel.local")
    )
    payload = {"risk": 80}
    result = client.submit_risk_result(payload, "token")

    assert result == {"accepted": True}
    assert captured["url"] == "https://tel.local/risk-results"
    assert captured["headers"]["Authorization"] == "Bearer token"
    assert captured["json"] == payload


def test_non_object_json_raises():
    client = EndpointClient(
        EndpointConfig(policy_base_url="https://policy.local", telemetry_base_url="https://tel.local")
    )

    with pytest.raises(ValueError, match="did not return a JSON object"):
        client._require_json_object([1, 2], "policy")
