from __future__ import annotations

import os

from src.spdm_state_machine import SpdmStateMachine
from src.spdm_transport import SpdmLoopbackTransport, SpdmQemuTransport


def make_spdm_session(ca_cert_der: bytes) -> SpdmStateMachine:
    if os.environ.get("IATO_HW_MODE", "0") == "1":
        default_socket = getattr(SpdmQemuTransport, "DEFAULT_SOCKET", "/tmp/iato-spdm.sock")
        transport = SpdmQemuTransport(os.environ.get("IATO_SPDM_SOCKET", default_socket))
    else:
        transport = SpdmLoopbackTransport()
        for r in [b"VERSION", b"CAPS", b"ALGS", b"ATTEST"]:
            transport.send(r)
    sm = SpdmStateMachine(ca_cert_der=ca_cert_der, transport=transport)
    sm.run_full_handshake()
    return sm
