#!/usr/bin/env python3
"""Generate ARMv9-A RME compliance evidence artifacts from model metadata."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class ControlEvidence:
    framework: str
    control_id: str
    requirement: str
    model_element: str
    proof_reference: str
    status: str = "satisfied"


EVIDENCE: list[ControlEvidence] = [
    ControlEvidence(
        framework="ARM CCA / RME",
        control_id="RMI_DATA_CREATE_AUTHZ",
        requirement="Data granules can only be created for active realms from delegated granules.",
        model_element="step / RmiCall.DataCreate",
        proof_reference="wf_dataCreate",
    ),
    ControlEvidence(
        framework="ARM CCA / RME",
        control_id="RMI_REALM_DESTROY_SCRUB",
        requirement="Realm destroy invalidates realm-owned/data granules.",
        model_element="realmDestroyGranule",
        proof_reference="realmDestroy_no_data_owner",
    ),
    ControlEvidence(
        framework="SOC2",
        control_id="CC6.1",
        requirement="Logical access and privilege boundaries are enforced.",
        model_element="RmeState.wf",
        proof_reference="wf_step",
    ),
    ControlEvidence(
        framework="ISM",
        control_id="0460",
        requirement="Privileged domain separation controls are verifiable.",
        model_element="GranuleState + RmiCall",
        proof_reference="IATO.V7.RMEModel",
    ),
]


def to_markdown(rows: list[ControlEvidence], generated_at: str) -> str:
    lines = [
        "# ARMv9-A RME Compliance Evidence",
        "",
        f"Generated at: `{generated_at}`",
        "",
        "| Framework | Control | Requirement | Model Element | Proof Reference | Status |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.framework} | {row.control_id} | {row.requirement} | "
            f"`{row.model_element}` | `{row.proof_reference}` | {row.status} |"
        )
    lines.append("")
    lines.append("Evidence source: `IATO_V7/IATO/V7/RMEModel.lean`.")
    return "\n".join(lines)


def main() -> None:
    now = datetime.now(timezone.utc).isoformat()
    out_dir = Path("artifacts/compliance")
    out_dir.mkdir(parents=True, exist_ok=True)

    json_out = out_dir / "armv9_rme_evidence.json"
    md_out = out_dir / "armv9_rme_evidence.md"

    payload = {
        "generated_at": now,
        "schema_version": "1.0",
        "evidence": [asdict(row) for row in EVIDENCE],
    }

    json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md_out.write_text(to_markdown(EVIDENCE, now) + "\n", encoding="utf-8")

    print(f"Wrote {json_out}")
    print(f"Wrote {md_out}")


if __name__ == "__main__":
    main()
