#!/usr/bin/env python3
"""Build deterministic release assets for the IATO-V7 Nmap path auditor."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python <3.11 fallback
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[3]
LAKEFILE = REPO_ROOT / "lakefile.toml"
ORCHESTRATOR = REPO_ROOT / "lean/iato_v7/scripts/nmap_path_audit_orchestrator.py"
RELEASE_DIR = REPO_ROOT / "lean/iato_v7/release"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_version() -> tuple[str, str]:
    data = tomllib.loads(LAKEFILE.read_text(encoding="utf-8"))
    project_version = str(data.get("version", "0.0.0"))
    nmap_cfg = data.get("tool", {}).get("iato_nmap", {})
    schema_version = str(nmap_cfg.get("schema_version", "1.0.0"))
    return project_version, schema_version


def build_release_python_script(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(ORCHESTRATOR.read_bytes())
    out_path.chmod(0o755)


def write_release_manifest(version: str, schema: str, files: list[Path]) -> None:
    entries = [
        {"name": file.name, "size": file.stat().st_size, "sha256": sha256_file(file)}
        for file in files
    ]

    release_manifest = {
        "created_utc": datetime.now(tz=timezone.utc).isoformat(),
        "project_version": version,
        "schema_version": schema,
        "assets": sorted(entries, key=lambda item: item["name"]),
    }
    (RELEASE_DIR / "release-manifest.json").write_text(
        json.dumps(release_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    checksums = "\n".join(
        f"{entry['sha256']}  {entry['name']}" for entry in sorted(entries, key=lambda item: item["name"])
    )
    (RELEASE_DIR / "SHA256SUMS").write_text(f"{checksums}\n", encoding="utf-8")


def main() -> int:
    version, schema = load_version()
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)

    release_script = RELEASE_DIR / f"iato-v7-nmap-audit-{version}.py"
    tar_gz = RELEASE_DIR / f"iato-v7-nmap-audit-{version}-linux-x86_64.tar.gz"

    build_release_python_script(release_script)
    if tar_gz.exists():
        tar_gz.unlink()

    write_release_manifest(version, schema, [release_script])

    print(f"[release] wrote {release_script}")
    print(f"[release] wrote {RELEASE_DIR / 'SHA256SUMS'}")
    print(f"[release] wrote {RELEASE_DIR / 'release-manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
