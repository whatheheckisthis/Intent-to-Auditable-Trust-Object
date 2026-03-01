#!/usr/bin/env python3
"""OSINT IAM parser using PolicyUniverse wildcard expansion."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

CRITICAL_ACTIONS = {
    "iam:createaccesskey",
}


def normalize_statements(policy: dict[str, Any]) -> list[dict[str, Any]]:
    statements = policy.get("Statement", [])
    if isinstance(statements, dict):
        statements = [statements]
    if not isinstance(statements, list):
        return []
    return [s for s in statements if isinstance(s, dict)]


def principal_has_wildcard(principal: Any) -> bool:
    if isinstance(principal, str):
        return "*" in principal
    if isinstance(principal, list):
        return any(principal_has_wildcard(p) for p in principal)
    if isinstance(principal, dict):
        return any(principal_has_wildcard(v) for v in principal.values())
    return False


def extract_actions(statement: dict[str, Any]) -> list[str]:
    raw = statement.get("Action", [])
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    return [a.lower() for a in raw if isinstance(a, str)]


def analyze(policy: dict[str, Any]) -> dict[str, Any]:
    try:
        from policyuniverse.expander import expand_action
    except Exception as exc:
        raise RuntimeError("policyuniverse dependency is required") from exc

    expanded_actions: set[str] = set()
    risks: list[str] = []

    for stmt in normalize_statements(policy):
        if stmt.get("Effect", "Allow") != "Allow":
            continue

        raw_actions = extract_actions(stmt)
        for action in raw_actions:
            for expanded in expand_action(action):
                expanded_actions.add(str(expanded).lower())

        if "sts:assumerole" in raw_actions and principal_has_wildcard(stmt.get("Principal")):
            risks.append("Critical: sts:AssumeRole allowed with wildcard principal.")

    critical_hits = sorted(expanded_actions & CRITICAL_ACTIONS)
    if critical_hits:
        risks.append(
            "Critical: high-risk actions present: " + ", ".join(critical_hits)
        )

    score = min(100, (40 if risks else 0) + (30 if "*" in expanded_actions else 0) + 30 * len(critical_hits))

    return {
        "risk_score": score,
        "expanded_actions_count": len(expanded_actions),
        "critical_risks": risks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to IAM policy JSON file")
    args = parser.parse_args()

    try:
        with open(args.file, "r", encoding="utf-8") as f:
            policy = json.load(f)
        if not isinstance(policy, dict):
            raise ValueError("Policy must be a JSON object")

        result = analyze(policy)
        sys.stdout.write(json.dumps(result, separators=(",", ":")))
        return 0
    except Exception as exc:  # JSON-only output contract
        sys.stdout.write(json.dumps({"error": str(exc)}, separators=(",", ":")))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
