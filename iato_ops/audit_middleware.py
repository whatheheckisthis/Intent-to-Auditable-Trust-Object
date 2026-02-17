"""Tamper-evident audit trail with RFC5424/syslog-ng forwarding."""

from __future__ import annotations

import hashlib
import json
import logging
import socket
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class AuditRecord:
    event: str
    actor: str
    session_id: str
    public_key_fingerprint: str
    details: dict[str, object]


class HashChainedAuditLogger:
    """Audit logger that links each event hash to the previous entry."""

    def __init__(self, app_name: str, syslog_host: str, syslog_port: int, hostname: str = "sdn-controller") -> None:
        self.app_name = app_name
        self.syslog_host = syslog_host
        self.syslog_port = syslog_port
        self.hostname = hostname
        self.previous_hash = "0" * 64
        self._local_logger = logging.getLogger("iato.audit")

    def emit(self, record: AuditRecord) -> str:
        timestamp = datetime.now(timezone.utc).isoformat()
        payload = {
            "timestamp": timestamp,
            "event": record.event,
            "actor": record.actor,
            "session_id": record.session_id,
            "public_key_fingerprint": record.public_key_fingerprint,
            "details": record.details,
            "previous_hash": self.previous_hash,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        entry_hash = hashlib.sha256(canonical.encode()).hexdigest()
        payload["entry_hash"] = entry_hash
        self.previous_hash = entry_hash

        syslog_message = self._to_rfc5424(payload)
        self._send_udp(syslog_message)
        self._local_logger.info(syslog_message)
        return entry_hash

    def _to_rfc5424(self, payload: dict[str, object]) -> str:
        pri = 14
        ts = payload["timestamp"]
        msgid = str(payload["event"]).upper().replace(" ", "_")
        structured_data = (
            f"[iato@32473 session_id=\"{payload['session_id']}\" "
            f"fingerprint=\"{payload['public_key_fingerprint']}\" "
            f"entry_hash=\"{payload['entry_hash']}\"]"
        )
        message = json.dumps(payload, sort_keys=True)
        return f"<{pri}>1 {ts} {self.hostname} {self.app_name} - {msgid} {structured_data} {message}"

    def _send_udp(self, message: str) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.sendto(message.encode(), (self.syslog_host, self.syslog_port))
