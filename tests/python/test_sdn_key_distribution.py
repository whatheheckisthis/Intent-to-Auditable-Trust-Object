"""Unit tests for SDN key distribution modules."""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from iato_ops.audit_middleware import AuditRecord, HashChainedAuditLogger
from iato_ops.hsm_pkcs11_wrapper import PKCS11Wrapper
from iato_ops.network_provisioning import RestconfTarget, SDNProvisioningWorkflow
from iato_ops.rbac import UserContext


class TestRBACAndWorkflow(unittest.TestCase):
    def setUp(self) -> None:
        self.user = UserContext(username="alice", role="Security_Admin", hsm_pin="123456")
        self.hsm = PKCS11Wrapper(expected_user_pin="123456")
        self.audit = HashChainedAuditLogger("sdn-keydist", "127.0.0.1", 514)
        self.workflow = SDNProvisioningWorkflow(self.hsm, self.audit)

    @patch("iato_ops.audit_middleware.HashChainedAuditLogger._send_udp")
    def test_rotate_key_captures_session_and_fingerprint(self, _mock_udp: Mock) -> None:
        result = self.workflow.rotate_key(
            user=self.user,
            token_label="token-1",
            key_label="switch01",
        )

        self.assertTrue(result["session_id"].startswith("sim-"))
        self.assertEqual(len(result["fingerprint"]), 64)

    def test_rotate_key_rejects_wrong_role(self) -> None:
        bad_user = UserContext(username="bob", role="Operator", hsm_pin="123456")

        with self.assertRaises(PermissionError):
            self.workflow.rotate_key(user=bad_user, token_label="token-1", key_label="switch01")

    @patch("iato_ops.network_provisioning.requests.put")
    @patch("iato_ops.audit_middleware.HashChainedAuditLogger._send_udp")
    def test_provision_node_uses_rfc8040_restconf_path(self, _mock_udp: Mock, mock_put: Mock) -> None:
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_put.return_value = mock_response

        rotation = self.workflow.rotate_key(
            user=self.user,
            token_label="token-1",
            key_label="switch01",
        )
        target = RestconfTarget("10.0.0.2", 443, "netops", "password")
        self.workflow.provision_node(
            user=self.user,
            target=target,
            key_name="switch01",
            public_key_pem=rotation["public_key_pem"],
            fingerprint=rotation["fingerprint"],
        )

        called_url = mock_put.call_args.args[0]
        self.assertIn("/restconf/data/ietf-keystore:keystore/asymmetric-keys/asymmetric-key=switch01", called_url)


class TestAuditChain(unittest.TestCase):
    @patch("iato_ops.audit_middleware.HashChainedAuditLogger._send_udp")
    def test_hash_chain_changes_per_entry(self, _mock_udp: Mock) -> None:
        logger = HashChainedAuditLogger("sdn-keydist", "127.0.0.1", 514)

        one = logger.emit(
            AuditRecord("rotate_key", "alice", "sess-1", "abc123", {"node": "sw1"})
        )
        two = logger.emit(
            AuditRecord("provision_node", "alice", "sess-1", "abc123", {"node": "sw1"})
        )

        self.assertNotEqual(one, two)


if __name__ == "__main__":
    unittest.main()
