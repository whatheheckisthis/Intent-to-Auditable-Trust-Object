#!/usr/bin/env python3
"""Deterministic Nmap orchestrator for host-path state collection.

This wrapper builds an Nmap command from repository templates and runtime
parameters, then executes `nmap -Pn -sn` with a custom NSE script to emit XML.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python <3.11 fallback
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TOML = REPO_ROOT / "lakefile.toml"
DEFAULT_SETUP_MD = REPO_ROOT / "lean/iato_v7/ci-proxy-worker-setup.md"
DEFAULT_K8S_YAML = REPO_ROOT / "lean/iato_v7/k8s-ci-proxy-worker.yaml"
DEFAULT_PROXY_SCRIPT = REPO_ROOT / "lean/iato_v7/run-ci-proxy.sh"
DEFAULT_POLICY_OUT = REPO_ROOT / "lean/iato_v7/.nmap-path-policy.json"


class ConfigError(RuntimeError):
    pass


def _read(path: Path) -> str:
    if not path.exists():
        raise ConfigError(f"required file missing: {path}")
    return path.read_text(encoding="utf-8")


def load_lake_template(path: Path) -> dict[str, Any]:
    data = tomllib.loads(_read(path))
    tool = data.get("tool", {})
    iato = tool.get("iato_nmap", {})
    if not iato:
        raise ConfigError("[tool.iato_nmap] not found in lakefile.toml")
    return {
        "project_name": data.get("name", "iato-v7"),
        "project_version": data.get("version", "unknown"),
        "nmap": iato,
    }


def parse_k8s_defaults(path: Path) -> dict[str, str]:
    text = _read(path)
    working_dir = re.search(r"workingDir:\s*(\S+)", text)
    scaffold = re.search(
        r"-\s+name:\s+SCAFFOLD_FILE\s+\n\s+value:\s*(\S+)", text, re.MULTILINE
    )
    return {
        "working_dir": working_dir.group(1) if working_dir else "/workspace",
        "scaffold_file": scaffold.group(1) if scaffold else "/config/k8s-build-job.yaml",
    }


def parse_proxy_defaults(path: Path) -> dict[str, str]:
    text = _read(path)
    workdir = re.search(r'WORKDIR="\$\{WORKDIR:-([^}]+)\}"', text)
    scaffold = re.search(r'SCAFFOLD_FILE="\$\{SCAFFOLD_FILE:-([^}]+)\}"', text)
    return {
        "workdir": workdir.group(1) if workdir else "/workspace",
        "scaffold_file": scaffold.group(1) if scaffold else "/config/k8s-build-job.yaml",
    }


def parse_setup_notes(path: Path) -> dict[str, str]:
    text = _read(path)
    mount_match = re.search(r"minikube mount[^\n]+", text)
    return {"mount_hint": mount_match.group(0) if mount_match else "minikube mount <src>:<dst>"}


def check_nmap_version(expected: str) -> None:
    proc = subprocess.run(["nmap", "--version"], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise ConfigError("nmap is required but not available in PATH")
    first_line = proc.stdout.splitlines()[0] if proc.stdout else ""
    found = re.search(r"([0-9]+\.[0-9]+)", first_line)
    if not found:
        raise ConfigError(f"unable to parse nmap version from: {first_line!r}")
    if found.group(1) != expected:
        raise ConfigError(f"determinism gate failed: expected nmap {expected}, found {found.group(1)}")


def validate_headless(nmap_cfg: dict[str, Any]) -> None:
    if nmap_cfg.get("headless", True) is not True:
        raise ConfigError("determinism gate failed: [tool.iato_nmap].headless must be true")


def build_policy(rules: list[dict[str, Any]], root_path: str) -> dict[str, Any]:
    ordered = sorted(rules, key=lambda item: item["path"])
    return {
        "root_path": root_path,
        "rules": ordered,
    }


def build_command(cfg: dict[str, Any], policy_path: Path, xml_out: Path) -> list[str]:
    nmap_cfg = cfg["nmap"]
    script_path = nmap_cfg["nse_script"]
    target = nmap_cfg.get("target", "127.0.0.1")
    script_args = {
        "path_audit.root": nmap_cfg["root_path"],
        "path_audit.policy": str(policy_path),
        "path_audit.schema": nmap_cfg.get("schema_version", "1.0.0"),
        "path_audit.project": cfg["project_name"],
        "path_audit.release": cfg["project_version"],
    }
    args_value = ",".join(f"{k}={script_args[k]}" for k in sorted(script_args.keys()))
    return [
        "nmap",
        "--noninteractive",
        "-Pn",
        "-sn",
        "-n",
        "--script",
        script_path,
        "--script-args",
        args_value,
        "-oX",
        str(xml_out),
        target,
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and run deterministic Nmap path-audit command")
    parser.add_argument("--config", type=Path, default=DEFAULT_TOML)
    parser.add_argument("--setup-md", type=Path, default=DEFAULT_SETUP_MD)
    parser.add_argument("--k8s-yaml", type=Path, default=DEFAULT_K8S_YAML)
    parser.add_argument("--proxy-script", type=Path, default=DEFAULT_PROXY_SCRIPT)
    parser.add_argument("--policy-out", type=Path, default=DEFAULT_POLICY_OUT)
    parser.add_argument("--xml-out", type=Path, default=REPO_ROOT / "lean/iato_v7/nmap-path-state.xml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-version-check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_lake_template(args.config)
    notes = parse_setup_notes(args.setup_md)
    k8s_defaults = parse_k8s_defaults(args.k8s_yaml)
    proxy_defaults = parse_proxy_defaults(args.proxy_script)

    nmap_cfg = cfg["nmap"]
    validate_headless(nmap_cfg)
    if not args.skip_version_check:
        check_nmap_version(nmap_cfg["nmap_version"])

    policy = build_policy(nmap_cfg.get("path_rules", []), nmap_cfg["root_path"])
    args.policy_out.parent.mkdir(parents=True, exist_ok=True)
    args.policy_out.write_text(json.dumps(policy, indent=2, sort_keys=True), encoding="utf-8")

    cmd = build_command(cfg, args.policy_out, args.xml_out)
    print("[iato-v7] deterministic orchestration context:")
    print(f"  mount_hint={notes['mount_hint']}")
    print(f"  k8s_working_dir={k8s_defaults['working_dir']}")
    print(f"  proxy_workdir={proxy_defaults['workdir']}")
    print(f"  policy_file={args.policy_out}")
    print(f"  xml_output={args.xml_out}")
    print(f"  command={shlex.join(cmd)}")

    if args.dry_run:
        return 0

    env = os.environ.copy()
    env["IATO_V7_STATELESS"] = "1"
    proc = subprocess.run(cmd, env=env, check=False)
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
