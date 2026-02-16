"""Endpoint client primitives for integrating policy and telemetry services."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class EndpointConfig:
    """Endpoint URLs and path segments for service integration."""

    policy_base_url: str
    telemetry_base_url: str
    policy_endpoint_prefix: str = "/policies"
    telemetry_risk_results_path: str = "/risk-results"
    timeout_seconds: int = 10


class EndpointClient:
    """Minimal endpoint client for JSON APIs used by the dispatcher toolchain."""

    def __init__(self, config: EndpointConfig) -> None:
        self.config = config

    def fetch_policy_document(self, policy_id: str, api_key: str) -> dict[str, Any]:
        response = requests.get(
            self._build_policy_url(policy_id),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=self.config.timeout_seconds,
        )
        response.raise_for_status()
        return self._require_json_object(response.json(), "policy")

    def submit_risk_result(self, payload: dict[str, Any], api_key: str) -> dict[str, Any]:
        response = requests.post(
            self._build_telemetry_url(),
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
            timeout=self.config.timeout_seconds,
        )
        response.raise_for_status()
        return self._require_json_object(response.json(), "telemetry")

    def _build_policy_url(self, policy_id: str) -> str:
        base = self.config.policy_base_url.rstrip("/")
        prefix = self.config.policy_endpoint_prefix.strip("/")
        return f"{base}/{prefix}/{policy_id}"

    def _build_telemetry_url(self) -> str:
        base = self.config.telemetry_base_url.rstrip("/")
        path = self.config.telemetry_risk_results_path.strip("/")
        return f"{base}/{path}"

    def _require_json_object(self, value: Any, endpoint_name: str) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError(f"{endpoint_name} endpoint did not return a JSON object")
        return value
