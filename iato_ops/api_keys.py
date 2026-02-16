"""API key loading utilities for production endpoint integrations."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class APIKeyConfig:
    """Holds API key environment variable names."""

    policy_service_key_env: str = "IATO_POLICY_SERVICE_API_KEY"
    telemetry_service_key_env: str = "IATO_TELEMETRY_API_KEY"


class APIKeyManager:
    """Loads API keys from environment variables without storing plaintext in files."""

    def __init__(self, config: APIKeyConfig | None = None) -> None:
        self.config = config or APIKeyConfig()

    def get_policy_service_key(self) -> str:
        return self._read_key(self.config.policy_service_key_env)

    def get_telemetry_service_key(self) -> str:
        return self._read_key(self.config.telemetry_service_key_env)

    def _read_key(self, env_name: str) -> str:
        value = os.getenv(env_name, "").strip()
        if not value:
            raise ValueError(f"Missing required API key in environment variable: {env_name}")
        return value
