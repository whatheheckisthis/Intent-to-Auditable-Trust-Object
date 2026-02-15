"""Validation tests for the minimal IATO evidence record schema."""

from datetime import datetime

REQUIRED_FIELDS = {
    "control_id",
    "service",
    "environment",
    "timestamp_utc",
    "source",
    "artifact_uri",
    "hash_sha256",
    "mapped_controls",
}


def validate_evidence_record(record: dict) -> None:
    missing = REQUIRED_FIELDS - set(record)
    assert not missing, f"missing required fields: {sorted(missing)}"

    timestamp = record["timestamp_utc"]
    assert isinstance(timestamp, str) and timestamp.endswith("Z"), "timestamp_utc must be an ISO8601 UTC string"
    # Convert `Z` to explicit UTC offset for parsing.
    datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

    mapped_controls = record["mapped_controls"]
    assert isinstance(mapped_controls, str) and mapped_controls.strip(), "mapped_controls must be a non-empty string"
    prefixes = ("SOC2:", "ASVS:", "E8:")
    assert any(prefix in mapped_controls for prefix in prefixes), (
        "mapped_controls should include at least one SOC2/ASVS/E8 reference"
    )

    if "exception_id" in record:
        assert "approver" in record and record["approver"], "approver is required when exception_id is set"


def test_accepts_baseline_evidence_record():
    record = {
        "control_id": "CTRL-IAM-006",
        "service": "payments-api",
        "environment": "prod",
        "timestamp_utc": "2026-02-15T02:30:00Z",
        "source": "ci/tools/run_all_checks.sh",
        "artifact_uri": "s3://audit-bucket/evidence/packet.json",
        "hash_sha256": "a" * 64,
        "mapped_controls": "SOC2:CC6.1, ASVS:V2, E8:MFA",
    }

    validate_evidence_record(record)


def test_accepts_exception_record_with_approver():
    record = {
        "control_id": "CTRL-IAM-006",
        "service": "payments-api",
        "environment": "prod",
        "timestamp_utc": "2026-02-15T02:30:00Z",
        "source": "ci/tools/run_all_checks.sh",
        "artifact_uri": "s3://audit-bucket/evidence/packet.json",
        "hash_sha256": "a" * 64,
        "mapped_controls": "SOC2:CC6.1, ASVS:V2, E8:MFA",
        "exception_id": "EXC-2026-0042",
        "approver": "security-oncall",
    }

    validate_evidence_record(record)


def test_rejects_exception_without_approver():
    record = {
        "control_id": "CTRL-IAM-006",
        "service": "payments-api",
        "environment": "prod",
        "timestamp_utc": "2026-02-15T02:30:00Z",
        "source": "ci/tools/run_all_checks.sh",
        "artifact_uri": "s3://audit-bucket/evidence/packet.json",
        "hash_sha256": "a" * 64,
        "mapped_controls": "SOC2:CC6.1, ASVS:V2, E8:MFA",
        "exception_id": "EXC-2026-0042",
    }

    try:
        validate_evidence_record(record)
    except AssertionError as exc:
        assert "approver is required" in str(exc)
    else:
        raise AssertionError("Expected validation failure when exception_id is set without approver")
