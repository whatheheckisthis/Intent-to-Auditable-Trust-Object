"""API key loading utilities for production endpoint integrations."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class APIKeyConfig:
    """Configuration for sourcing API keys from environment and AWS Secrets Manager."""

    policy_service_key_env: str = "IATO_POLICY_SERVICE_API_KEY"
    telemetry_service_key_env: str = "IATO_TELEMETRY_API_KEY"
    aws_region_env: str = "AWS_REGION"
    aws_secret_id_env: str = "IATO_API_KEYS_SECRET_ID"
    aws_policy_secret_field: str = "policy_service_api_key"
    aws_telemetry_secret_field: str = "telemetry_service_api_key"


class APIKeyManager:
    """Load API keys from AWS Secrets Manager first, then environment fallback."""

    def __init__(self, config: APIKeyConfig | None = None) -> None:
        self.config = config or APIKeyConfig()

    def get_policy_service_key(self) -> str:
        secret_value = self._read_key_from_aws(self.config.aws_policy_secret_field)
        if secret_value:
            return secret_value
        return self._read_key_from_env(self.config.policy_service_key_env)

    def get_telemetry_service_key(self) -> str:
        secret_value = self._read_key_from_aws(self.config.aws_telemetry_secret_field)
        if secret_value:
            return secret_value
        return self._read_key_from_env(self.config.telemetry_service_key_env)

    def _read_key_from_env(self, env_name: str) -> str:
        value = os.getenv(env_name, "").strip()
        if not value:
            raise ValueError(f"Missing required API key in environment variable: {env_name}")
        return value

    def _read_key_from_aws(self, field_name: str) -> str | None:
        secret_id = os.getenv(self.config.aws_secret_id_env, "").strip()
        if not secret_id:
            return None

        region_name = os.getenv(self.config.aws_region_env, "").strip() or None

        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError(
                "boto3 is required to load API keys from AWS Secrets Manager"
            ) from exc

        client = boto3.client("secretsmanager", region_name=region_name)
        response: dict[str, Any] = client.get_secret_value(SecretId=secret_id)
        secret_string = str(response.get("SecretString", "")).strip()
        if not secret_string:
            return None

        secret_payload = json.loads(secret_string)
        if not isinstance(secret_payload, dict):
            raise ValueError("AWS secret payload must be a JSON object")

        value = str(secret_payload.get(field_name, "")).strip()
        return value or None
