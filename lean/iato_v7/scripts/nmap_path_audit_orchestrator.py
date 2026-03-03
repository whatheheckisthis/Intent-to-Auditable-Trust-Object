#!/usr/bin/env python3
"""IĀTŌ‑V7 deterministic Nmap orchestrator."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "lean/iato_v7/config.toml"
DEFAULT_XML_OUT = REPO_ROOT / "lean/iato_v7/nmap-path-state.xml"


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class HostProfile:
    environment: str
    timing_template: str


def _read(path: Path) -> str:
    if not path.exists():
        raise ConfigError(f"required file missing: {path}")
    return path.read_text(encoding="utf-8")


def _require_keys(mapping: dict[str, Any], keys: list[str], prefix: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ConfigError(f"{prefix} missing required keys: {', '.join(missing)}")


def load_config(path: Path) -> dict[str, Any]:
    data = tomllib.loads(_read(path))
    orchestrator = data.get("orchestrator", {})
    nmap_cfg = data.get("nmap", {})
    nse_cfg = data.get("nse", {})
    schema_cfg = data.get("schema", {})

    _require_keys(orchestrator, ["project", "release"], "[orchestrator]")
    _require_keys(
        nmap_cfg,
        ["binary", "target", "script", "xml_output", "timing", "performance", "flags"],
        "[nmap]",
    )
    _require_keys(nse_cfg, ["root_path", "rules", "enforce_strict"], "[nse]")
    _require_keys(schema_cfg, ["expected_version"], "[schema]")

    return {"orchestrator": orchestrator, "nmap": nmap_cfg, "nse": nse_cfg, "schema": schema_cfg}


def detect_host_profile(timing_cfg: dict[str, Any]) -> HostProfile:
    env_name = timing_cfg.get("default_env", "linux")
    if "WSL_DISTRO_NAME" in os.environ:
        env_name = "wsl2"
    elif "KUBERNETES_SERVICE_HOST" in os.environ:
        env_name = "kubernetes"

    templates = timing_cfg.get("templates", {})
    timing_template = templates.get(env_name) or templates.get("linux") or "T3"
    if not re.fullmatch(r"T[0-5]", timing_template):
        raise ConfigError(f"invalid timing template {timing_template!r}; expected T0..T5")
    return HostProfile(environment=env_name, timing_template=timing_template)


def serialize_rules(rules: list[dict[str, Any]]) -> str:
    encoded_rows: list[str] = []
    for rule in rules:
        row = [
            str(rule.get("path", "-")),
            str(rule.get("permissions", "-")),
            str(rule.get("uid", "-")),
            str(rule.get("gid", "-")),
            str(rule.get("sha256", "-")),
            str(rule.get("size", "-")),
        ]
        if any("," in col or "~" in col or ";" in col for col in row):
            raise ConfigError("nse rule values may not contain ',', '~', or ';'")
        encoded_rows.append("~".join(row))
    return ";".join(encoded_rows)




def ensure_binary(binary: str) -> None:
    if not shutil.which(binary):
        raise ConfigError(f"required binary not found in PATH: {binary}")

def build_command(cfg: dict[str, Any], xml_out: Path, profile: HostProfile) -> list[str]:
    nmap_cfg = cfg["nmap"]
    flags = nmap_cfg.get("flags", [])
    if "-Pn" not in flags or "-sn" not in flags:
        raise ConfigError("[nmap].flags must include -Pn and -sn")

    nse_cfg = cfg["nse"]
    script_args = {
        "path_audit.root": nse_cfg["root_path"],
        "path_audit.schema": cfg["schema"]["expected_version"],
        "path_audit.project": cfg["orchestrator"]["project"],
        "path_audit.release": cfg["orchestrator"]["release"],
        "path_audit.strict": "1" if nse_cfg.get("enforce_strict", True) else "0",
        "path_audit.rules": serialize_rules(nse_cfg.get("rules", [])),
    }
    script_args_str = ",".join(f"{k}={script_args[k]}" for k in sorted(script_args))

    command = [nmap_cfg.get("binary", "nmap"), "--noninteractive", *flags, "-n", f"-{profile.timing_template}"]
    for perf_flag in nmap_cfg.get("performance", []):
        command.extend(shlex.split(perf_flag))

    command.extend([
        "--script",
        nmap_cfg["script"],
        "--script-args",
        script_args_str,
        "-oX",
        str(xml_out),
        nmap_cfg["target"],
    ])
    return command


def validate_iato_xml(path: Path, expected_schema: str) -> tuple[bool, str]:
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as err:
        return False, f"XML parse failure: {err}"

    if root.tag != "nmaprun":
        return False, "schema violation: root element must be <nmaprun>"

    scripts = root.findall("./host/hostscript/script")
    if not scripts:
        return False, "schema violation: missing /nmaprun/host/hostscript/script"

    payload = "\n".join(script.attrib.get("output", "") for script in scripts)
    schema_match = re.search(r"schema=([0-9]+\.[0-9]+\.[0-9]+)", payload)
    if not schema_match:
        return False, "schema violation: missing schema marker from NSE output"
    if schema_match.group(1) != expected_schema:
        return False, f"schema violation: expected schema {expected_schema}, got {schema_match.group(1)}"

    violation_match = re.search(r"violation_count=(\d+)", payload)
    if not violation_match:
        return False, "schema violation: missing violation_count marker from NSE output"
    if int(violation_match.group(1)) != 0:
        return False, f"NSE integrity violations found: {violation_match.group(1)}"
    return True, "IATO-V7 XML validation passed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic Nmap file-system audit")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--xml-out", type=Path, default=DEFAULT_XML_OUT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    profile = detect_host_profile(cfg["nmap"]["timing"])

    xml_out = args.xml_out if args.xml_out != DEFAULT_XML_OUT else Path(cfg["nmap"]["xml_output"])
    xml_out.parent.mkdir(parents=True, exist_ok=True)

    command = build_command(cfg, xml_out, profile)
    print("[iato-v7] orchestrator profile")
    print(f"  env={profile.environment}")
    print(f"  timing={profile.timing_template}")
    print(f"  xml_output={xml_out}")
    print(f"  command={shlex.join(command)}")

    if args.dry_run:
        return 0

    ensure_binary(cfg["nmap"]["binary"])
    proc = subprocess.run(command, check=False)
    if proc.returncode != 0:
        return proc.returncode

    ok, message = validate_iato_xml(xml_out, cfg["schema"]["expected_version"])
    print(f"[iato-v7] {message}")
    return 0 if ok else 3


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ConfigError as err:
        print(f"[iato-v7] configuration error: {err}", file=sys.stderr)
        sys.exit(2)
