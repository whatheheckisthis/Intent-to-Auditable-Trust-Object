"""Endpoint client primitives for integrating policy and telemetry services."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class EndpointConfig:
    """Endpoint base URLs for service integration."""

    policy_base_url: str
    telemetry_base_url: str
    timeout_seconds: int = 10


class EndpointClient:
    """Minimal endpoint client for JSON APIs used by the dispatcher toolchain."""

    def __init__(self, config: EndpointConfig) -> None:
        self.config = config

    def fetch_policy_document(self, policy_id: str, api_key: str) -> dict[str, Any]:
        response = requests.get(
            f"{self.config.policy_base_url.rstrip('/')}/policies/{policy_id}",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=self.config.timeout_seconds,
        )
        response.raise_for_status()
        return self._require_json_object(response.json(), "policy")

    def submit_risk_result(self, payload: dict[str, Any], api_key: str) -> dict[str, Any]:
        response = requests.post(
            f"{self.config.telemetry_base_url.rstrip('/')}/risk-results",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
            timeout=self.config.timeout_seconds,
        )
        response.raise_for_status()
        return self._require_json_object(response.json(), "telemetry")

    def _require_json_object(self, value: Any, endpoint_name: str) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError(f"{endpoint_name} endpoint did not return a JSON object")
        return value
