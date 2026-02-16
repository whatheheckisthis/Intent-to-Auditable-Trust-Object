import json
import types

import pytest

from iato_ops.iam_policy_parser import IAMParser, main


@pytest.fixture(autouse=True)
def fake_policyuniverse(monkeypatch):
    module = types.ModuleType("policyuniverse.expander")

    def expand_action(action):
        mapping = {
            "iam:*": ["iam:createaccesskey", "iam:updateassumerolepolicy"],
            "s3:*": ["s3:putbucketpolicy", "s3:getbucketlocation"],
        }
        return mapping.get(action.lower(), [action.lower()])

    module.expand_action = expand_action
    monkeypatch.setitem(__import__("sys").modules, "policyuniverse.expander", module)


def test_analyze_detects_admin_and_risks():
    policy = {
        "Statement": [
            {"Effect": "Allow", "Action": ["iam:*", "s3:*", "sts:AssumeRole"], "Principal": "*"}
        ]
    }

    result = IAMParser(policy).analyze()

    assert result["is_admin"] is True
    assert result["risk_score"] == 100
    assert len(result["risk_warnings"]) == 3


def test_main_with_policy_argument(monkeypatch, capsys):
    policy = {"Statement": {"Effect": "Allow", "Action": "s3:PutBucketPolicy"}}
    monkeypatch.setattr(
        "sys.argv",
        ["iam-policy-parser", "--policy", json.dumps(policy)],
    )

    code = main()
    out = capsys.readouterr().out

    assert code == 0
    parsed = json.loads(out)
    assert parsed["risk_score"] >= 30


def test_main_returns_error_for_invalid_json(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["iam-policy-parser", "--policy", "{"])

    code = main()
    out = capsys.readouterr().out

    assert code == 1
    assert "error" in json.loads(out)
