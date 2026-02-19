"""Secure SDN key distribution workflow using RBAC + PKCS#11 + RESTCONF."""

from __future__ import annotations

import base64
from importlib import import_module, util
import types
from dataclasses import dataclass

try:
    _requests_spec = util.find_spec("requests")
except ValueError:
    _requests_spec = None

requests = import_module("requests") if _requests_spec else types.SimpleNamespace(put=None, Response=object)

from iato_ops.audit_middleware import AuditRecord, HashChainedAuditLogger
from iato_ops.hsm_pkcs11_wrapper import PKCS11Wrapper
from iato_ops.rbac import UserContext, require_role


@dataclass(frozen=True)
class RestconfTarget:
    host: str
    port: int
    username: str
    password: str


class SDNProvisioningWorkflow:
    def __init__(self, hsm: PKCS11Wrapper, audit: HashChainedAuditLogger) -> None:
        self.hsm = hsm
        self.audit = audit

    @require_role("Security_Admin")
    def rotate_key(self, *, user: UserContext, token_label: str, key_label: str) -> dict[str, str]:
        session_id = self.hsm.open_session(token_label=token_label)
        public_key = self.hsm.generate_keypair(label=key_label)

        self.audit.emit(
            AuditRecord(
                event="rotate_key",
                actor=user.username,
                session_id=session_id,
                public_key_fingerprint=public_key.fingerprint,
                details={"key_label": key_label},
            )
        )
        return {"session_id": session_id, "public_key_pem": public_key.pem, "fingerprint": public_key.fingerprint}

    @require_role("Security_Admin")
    def provision_node(
        self,
        *,
        user: UserContext,
        target: RestconfTarget,
        key_name: str,
        public_key_pem: str,
        fingerprint: str,
    ) -> requests.Response:
        """Push public key to RESTCONF keystore per RFC 8040."""

        url = (
            f"https://{target.host}:{target.port}/restconf/data/"
            f"ietf-keystore:keystore/asymmetric-keys/asymmetric-key={key_name}"
        )
        body = {
            "ietf-keystore:asymmetric-key": {
                "name": key_name,
                "algorithm": "rsa3072",
                "public-key-format": "subject-public-key-info",
                "public-key": base64.b64encode(public_key_pem.encode()).decode(),
            }
        }
        if requests.put is None:
            raise RuntimeError(
                "requests dependency is unavailable; install project dependencies before provisioning"
            )

        response = requests.put(
            url,
            auth=(target.username, target.password),
            headers={
                "Accept": "application/yang-data+json",
                "Content-Type": "application/yang-data+json",
            },
            json=body,
            timeout=15,
            verify=False,
        )
        response.raise_for_status()

        self.audit.emit(
            AuditRecord(
                event="provision_node",
                actor=user.username,
                session_id=self.hsm.session_id or "unknown-session",
                public_key_fingerprint=fingerprint,
                details={"target": target.host, "restconf_path": url},
            )
        )
        return response


def run_provisioning_flow() -> None:
    """Reference workflow executable for operators/automation."""

    user = UserContext(username="alice", role="Security_Admin", hsm_pin="123456")
    hsm = PKCS11Wrapper(expected_user_pin="123456")
    audit = HashChainedAuditLogger(app_name="sdn-keydist", syslog_host="127.0.0.1", syslog_port=514)
    workflow = SDNProvisioningWorkflow(hsm=hsm, audit=audit)

    rotated = workflow.rotate_key(user=user, token_label="IATO-TOKEN", key_label="switch01-key")

    target = RestconfTarget(host="10.0.0.2", port=443, username="netops", password="strong-password")
    workflow.provision_node(
        user=user,
        target=target,
        key_name="switch01-key",
        public_key_pem=rotated["public_key_pem"],
        fingerprint=rotated["fingerprint"],
    )


if __name__ == "__main__":
    run_provisioning_flow()
