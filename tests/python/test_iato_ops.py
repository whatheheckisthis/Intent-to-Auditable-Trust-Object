"""Production-grade unit tests for iato_ops helpers and clients."""

from __future__ import annotations

import json
import sys
import types
import unittest
from unittest.mock import Mock, patch

# Ensure tests run in minimal environments where `requests` may not be installed.
requests_stub = types.SimpleNamespace(get=Mock(name="requests.get"), post=Mock(name="requests.post"))
sys.modules.setdefault("requests", requests_stub)

from iato_ops.api_keys import APIKeyManager
from iato_ops.endpoints import EndpointClient, EndpointConfig
from iato_ops.iam_policy_parser import IAMParser
from iato_ops.syscalls import collect_runtime_syscalls


class TestRuntimeSyscalls(unittest.TestCase):
    def test_collect_runtime_syscalls_contains_expected_shape(self) -> None:
        snapshot = collect_runtime_syscalls()

        self.assertGreater(snapshot["pid"], 0)
        self.assertGreaterEqual(snapshot["ppid"], 0)
        self.assertIsInstance(snapshot["cwd"], str)
        self.assertTrue(snapshot["cwd"])

        uname = snapshot["uname"]
        self.assertIsInstance(uname["system"], str)
        self.assertIsInstance(uname["release"], str)
        self.assertIsInstance(uname["version"], str)
        self.assertIsInstance(uname["machine"], str)
        self.assertIsInstance(snapshot["monotonic_ns"], int)
        self.assertIsInstance(snapshot["time_ns"], int)


class TestAPIKeyManager(unittest.TestCase):
    @patch.dict(
        "os.environ",
        {
            "IATO_POLICY_SERVICE_API_KEY": " policy-token ",
            "IATO_TELEMETRY_API_KEY": " telemetry-token ",
        },
        clear=True,
    )
    def test_getters_read_and_strip_env_values(self) -> None:
        manager = APIKeyManager()
        self.assertEqual(manager.get_policy_service_key(), "policy-token")
        self.assertEqual(manager.get_telemetry_service_key(), "telemetry-token")

    @patch.dict("os.environ", {}, clear=True)
    def test_missing_key_raises_value_error(self) -> None:
        manager = APIKeyManager()
        with self.assertRaisesRegex(ValueError, "IATO_TELEMETRY_API_KEY"):
            manager.get_telemetry_service_key()

    @patch.dict(
        "os.environ",
        {
            "AWS_REGION": "us-east-1",
            "IATO_API_KEYS_SECRET_ID": "prod/iato/api-keys",
        },
        clear=True,
    )
    def test_policy_key_prefers_aws_secret_manager(self) -> None:
        fake_client = Mock()
        fake_client.get_secret_value.return_value = {
            "SecretString": json.dumps(
                {
                    "policy_service_api_key": "aws-policy-key",
                    "telemetry_service_api_key": "aws-telemetry-key",
                }
            )
        }
        boto3_stub = types.SimpleNamespace(client=Mock(return_value=fake_client))

        with patch.dict(sys.modules, {"boto3": boto3_stub}):
            manager = APIKeyManager()
            self.assertEqual(manager.get_policy_service_key(), "aws-policy-key")
            self.assertEqual(manager.get_telemetry_service_key(), "aws-telemetry-key")

        boto3_stub.client.assert_called_with("secretsmanager", region_name="us-east-1")
        fake_client.get_secret_value.assert_called_with(SecretId="prod/iato/api-keys")


class TestEndpointClient(unittest.TestCase):
    def setUp(self) -> None:
        self.config = EndpointConfig(
            policy_base_url="https://api.iato.example.com",
            telemetry_base_url="https://telemetry.iato.example.com",
            policy_endpoint_prefix="/v1/policies",
            telemetry_risk_results_path="/v1/risk-results",
            timeout_seconds=15,
        )
        self.client = EndpointClient(self.config)

    @patch("iato_ops.endpoints.requests.get")
    def test_fetch_policy_document_uses_expected_contract(self, mock_get: Mock) -> None:
        mock_response = Mock()
        mock_response.json.return_value = {"policy_id": "abc-123"}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        payload = self.client.fetch_policy_document("abc-123", "token-1")

        self.assertEqual(payload, {"policy_id": "abc-123"})
        mock_get.assert_called_once_with(
            "https://api.iato.example.com/v1/policies/abc-123",
            headers={"Authorization": "Bearer token-1"},
            timeout=15,
        )

    @patch("iato_ops.endpoints.requests.post")
    def test_submit_risk_result_uses_expected_contract(self, mock_post: Mock) -> None:
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"status": "accepted"}
        mock_post.return_value = mock_response

        payload = {"risk_score": 90, "warnings": ["privilege escalation"]}
        result = self.client.submit_risk_result(payload, "token-2")

        self.assertEqual(result, {"status": "accepted"})
        mock_post.assert_called_once_with(
            "https://telemetry.iato.example.com/v1/risk-results",
            headers={"Authorization": "Bearer token-2"},
            json=payload,
            timeout=15,
        )


class TestIAMParser(unittest.TestCase):
    def test_analyze_detects_admin_from_wildcard_raw_pattern(self) -> None:
        parser = IAMParser({"Statement": [{"Effect": "Allow", "Action": "*"}]})

        with patch.object(IAMParser, "_expand_statement_actions", return_value=set()):
            result = parser.analyze()

        self.assertTrue(result["is_admin"])
        self.assertEqual(result["risk_score"], 30)

    def test_analyze_builds_composite_score_with_cap(self) -> None:
        policy = {
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": ["iam:CreateAccessKey", "s3:PutBucketPolicy", "sts:AssumeRole"],
                    "Principal": {"AWS": "*"},
                }
            ]
        }
        parser = IAMParser(policy)

        expanded = {"*", "iam:createaccesskey", "s3:putbucketpolicy"}
        with patch.object(IAMParser, "_expand_statement_actions", return_value=expanded):
            result = parser.analyze()

        self.assertTrue(result["is_admin"])
        self.assertEqual(result["risk_score"], 100)
        self.assertEqual(len(result["risk_warnings"]), 3)


if __name__ == "__main__":
    unittest.main()
