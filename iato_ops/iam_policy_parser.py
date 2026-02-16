#!/usr/bin/env python3
"""Production IAM policy parser for C-based dispatchers."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RiskWeights:
    """Risk score weights for critical findings."""

    privilege_escalation: int = 40
    data_exposure: int = 30
    insecure_trust: int = 40


class IAMParser:
    """Parse IAM policies and emit deterministic, JSON-safe risk output."""

    PRIV_ESCALATION_ACTIONS = {
        "iam:createaccesskey",
        "iam:updateassumerolepolicy",
    }
    DATA_EXPOSURE_ACTIONS = {
        "s3:putbucketpolicy",
        "s3:getbucketlocation",
    }

    def __init__(self, policy: dict[str, Any], weights: RiskWeights | None = None) -> None:
        if not isinstance(policy, dict):
            raise ValueError("Policy must be a JSON object.")
        self.policy = policy
        self.weights = weights or RiskWeights()

    def analyze(self) -> dict[str, Any]:
        statements = self._normalize_statements(self.policy.get("Statement", []))
        expanded_actions: set[str] = set()
        raw_action_patterns: set[str] = set()

        has_priv_escalation = False
        has_data_exposure = False
        has_insecure_trust = False

        for statement in statements:
            if statement.get("Effect", "Allow") != "Allow":
                continue

            raw_action_patterns.update(self._extract_raw_actions(statement))
            statement_actions = self._expand_statement_actions(statement)
            expanded_actions.update(statement_actions)

            has_priv_escalation = has_priv_escalation or bool(
                statement_actions & self.PRIV_ESCALATION_ACTIONS
            )
            has_data_exposure = has_data_exposure or bool(
                statement_actions & self.DATA_EXPOSURE_ACTIONS
            )
            has_insecure_trust = has_insecure_trust or self._has_insecure_trust(statement)

        is_admin = self._is_admin_policy(expanded_actions, raw_action_patterns)
        risk_warnings = self._build_risk_warnings(
            expanded_actions,
            has_priv_escalation,
            has_data_exposure,
            has_insecure_trust,
        )

        risk_score = self._build_risk_score(
            is_admin=is_admin,
            has_priv_escalation=has_priv_escalation,
            has_data_exposure=has_data_exposure,
            has_insecure_trust=has_insecure_trust,
        )

        return {
            "is_admin": is_admin,
            "risk_score": risk_score,
            "risk_warnings": risk_warnings,
        }

    def _normalize_statements(self, statements: Any) -> list[dict[str, Any]]:
        if isinstance(statements, dict):
            statements = [statements]
        if not isinstance(statements, list):
            raise ValueError("Policy Statement must be an object or array.")
        return [statement for statement in statements if isinstance(statement, dict)]

    def _extract_raw_actions(self, statement: dict[str, Any]) -> set[str]:
        raw_actions = statement.get("Action", [])
        actions = [raw_actions] if isinstance(raw_actions, str) else raw_actions
        if not isinstance(actions, list):
            return set()
        return {action.lower() for action in actions if isinstance(action, str)}

    def _expand_statement_actions(self, statement: dict[str, Any]) -> set[str]:
        actions = list(self._extract_raw_actions(statement))
        if not isinstance(actions, list):
            return set()

        try:
            from policyuniverse.expander import expand_action
        except Exception as exc:  # pylint: disable=broad-except
            raise RuntimeError(
                "policyuniverse dependency is required for wildcard expansion"
            ) from exc

        expanded: set[str] = set()
        for action in actions:
            if not isinstance(action, str):
                continue
            for resolved in expand_action(action):
                expanded.add(str(resolved).lower())
        return expanded

    def _has_insecure_trust(self, statement: dict[str, Any]) -> bool:
        actions = statement.get("Action", [])
        if isinstance(actions, str):
            actions = [actions]
        if not isinstance(actions, list):
            return False

        has_assume_role = any(
            isinstance(action, str) and action.lower() == "sts:assumerole"
            for action in actions
        )
        return has_assume_role and self._principal_contains_wildcard(statement.get("Principal"))

    def _principal_contains_wildcard(self, principal: Any) -> bool:
        if isinstance(principal, str):
            return "*" in principal
        if isinstance(principal, list):
            return any(self._principal_contains_wildcard(entry) for entry in principal)
        if isinstance(principal, dict):
            return any(self._principal_contains_wildcard(value) for value in principal.values())
        return False

    def _is_admin_policy(
        self,
        expanded_actions: set[str],
        raw_action_patterns: set[str],
    ) -> bool:
        return any(token in {"*", "iam:*"} for token in expanded_actions | raw_action_patterns)

    def _build_risk_warnings(
        self,
        expanded_actions: set[str],
        has_priv_escalation: bool,
        has_data_exposure: bool,
        has_insecure_trust: bool,
    ) -> list[str]:
        warnings: list[str] = []
        if has_priv_escalation:
            matched = sorted(expanded_actions & self.PRIV_ESCALATION_ACTIONS)
            warnings.append(
                f"Critical: Privilege escalation permissions present: {', '.join(matched)}."
            )
        if has_data_exposure:
            matched = sorted(expanded_actions & self.DATA_EXPOSURE_ACTIONS)
            warnings.append(
                f"Critical: Data exposure permissions present: {', '.join(matched)}."
            )
        if has_insecure_trust:
            warnings.append(
                "Critical: Insecure trust policy allows sts:AssumeRole with wildcard principal."
            )
        return warnings

    def _build_risk_score(
        self,
        *,
        is_admin: bool,
        has_priv_escalation: bool,
        has_data_exposure: bool,
        has_insecure_trust: bool,
    ) -> int:
        score = 0
        if is_admin:
            score += 30
        if has_priv_escalation:
            score += self.weights.privilege_escalation
        if has_data_exposure:
            score += self.weights.data_exposure
        if has_insecure_trust:
            score += self.weights.insecure_trust
        return min(100, score)


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="iam-policy-parser", add_help=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--policy", type=str, help="Raw IAM policy JSON string")
    source.add_argument("--file", type=str, help="Path to IAM policy JSON file")
    return parser


def _load_policy_from_args(args: argparse.Namespace) -> dict[str, Any]:
    if args.policy:
        return json.loads(args.policy)
    with open(args.file, "r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def main() -> int:
    try:
        parser = _build_cli_parser()
        args = parser.parse_args()
        policy = _load_policy_from_args(args)
        result = IAMParser(policy).analyze()
        sys.stdout.write(json.dumps(result, separators=(",", ":")))
        return 0
    except Exception as exc:  # pylint: disable=broad-except
        sys.stdout.write(json.dumps({"error": str(exc)}, separators=(",", ":")))
        return 1


if __name__ == "__main__":
    sys.exit(main())
