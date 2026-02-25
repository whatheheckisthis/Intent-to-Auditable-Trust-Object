#!/usr/bin/env python3
"""Dependency Management Engine for formal verification toolchain.

Features:
- Pre-flight checks for tlc/coqc and version validation.
- Network install wrapper around ./scripts/install-formal-verification-deps.sh.
- HTTP 403/407-aware fallback to LOCAL_ARTIFACT_ROOT.
- Proxy and custom CA propagation for subprocess + download requests.
- deps_manifest.json state tracking to avoid repeated failing attempts.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import ssl
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, build_opener, install_opener, Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALL_SCRIPT = REPO_ROOT / "scripts" / "install-formal-verification-deps.sh"
MANIFEST_PATH = REPO_ROOT / "deps_manifest.json"
TOOLS = ("tlc", "coqc")
TLA_JAR_URL = "https://github.com/tlaplus/tlaplus/releases/latest/download/tla2tools.jar"


@dataclass
class ToolState:
    present: bool
    path: Optional[str]
    version_output: Optional[str]


class DepEngine:
    def __init__(self, manifest_path: Path, install_script: Path) -> None:
        self.manifest_path = manifest_path
        self.install_script = install_script
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> Dict:
        if self.manifest_path.exists():
            try:
                return json.loads(self.manifest_path.read_text())
            except json.JSONDecodeError:
                return {"schema": 1, "history": []}
        return {"schema": 1, "history": []}

    def _save_manifest(self) -> None:
        self.manifest_path.write_text(json.dumps(self.manifest, indent=2, sort_keys=True) + "\n")

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _run(self, cmd: list[str], extra_env: Optional[Dict[str, str]] = None) -> subprocess.CompletedProcess:
        env = os.environ.copy()
        if extra_env:
            env.update({k: v for k, v in extra_env.items() if v is not None})
        return subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, env=env)

    def _probe_tool(self, tool: str) -> ToolState:
        path = shutil.which(tool)
        if not path:
            return ToolState(False, None, None)
        proc = self._run([tool, "--version"]) if tool == "coqc" else self._run([tool, "-version"])
        output = (proc.stdout + "\n" + proc.stderr).strip()
        return ToolState(proc.returncode == 0, path, output)

    def preflight(self) -> Dict[str, ToolState]:
        return {tool: self._probe_tool(tool) for tool in TOOLS}

    def all_tools_present(self, states: Dict[str, ToolState]) -> bool:
        return all(states[t].present for t in TOOLS)

    def _record_attempt(self, stage: str, success: bool, details: Dict) -> None:
        self.manifest.setdefault("history", []).append(
            {
                "time_utc": self._now(),
                "stage": stage,
                "success": success,
                "details": details,
            }
        )
        self._save_manifest()

    def _latest_failed_net_attempt(self) -> Optional[Dict]:
        for item in reversed(self.manifest.get("history", [])):
            if item.get("stage") == "network_install" and not item.get("success"):
                return item
        return None

    def should_skip_network(self) -> bool:
        last_failure = self._latest_failed_net_attempt()
        if not last_failure:
            return False
        if os.environ.get("LOCAL_ARTIFACT_ROOT"):
            return False
        return True

    def proxy_env(self) -> Dict[str, str]:
        env: Dict[str, str] = {}
        for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "no_proxy"):
            val = os.environ.get(key)
            if val:
                env[key] = val
        ca_bundle = os.environ.get("CUSTOM_CA_BUNDLE")
        if ca_bundle:
            env["CURL_CA_BUNDLE"] = ca_bundle
            env["SSL_CERT_FILE"] = ca_bundle
            env["REQUESTS_CA_BUNDLE"] = ca_bundle
        return env

    def configure_url_opener(self) -> ssl.SSLContext:
        proxies = {}
        if os.environ.get("http_proxy"):
            proxies["http"] = os.environ["http_proxy"]
        if os.environ.get("https_proxy"):
            proxies["https"] = os.environ["https_proxy"]
        opener = build_opener(ProxyHandler(proxies))
        install_opener(opener)

        ca_bundle = os.environ.get("CUSTOM_CA_BUNDLE")
        if ca_bundle:
            return ssl.create_default_context(cafile=ca_bundle)
        return ssl.create_default_context()

    def run_install_script(self) -> Tuple[bool, str, str]:
        env = self.proxy_env()
        proc = self._run([str(self.install_script)], extra_env=env)
        return proc.returncode == 0, proc.stdout, proc.stderr

    def _http_status_from_error(self, text: str) -> Optional[int]:
        if "403" in text:
            return 403
        if "407" in text:
            return 407
        return None

    def _suggest_proxy_flags(self) -> str:
        return (
            "Try one of:\n"
            "  - export LOCAL_ARTIFACT_ROOT=/path/to/pre-mirrored-artifacts\n"
            "  - export http_proxy=http://proxy:port && export https_proxy=http://proxy:port\n"
            "  - export NO_PROXY=localhost,127.0.0.1,<internal-domains>\n"
            "  - apt with explicit bypass (if policy allows): apt-get -o Acquire::http::Proxy=false -o Acquire::https::Proxy=false <cmd>"
        )

    def _download_tla_from_network(self, target_jar: Path) -> Tuple[bool, str]:
        ctx = self.configure_url_opener()
        try:
            req = Request(TLA_JAR_URL, headers={"User-Agent": "formal-deps-engine/1.0"})
            with urlopen(req, context=ctx, timeout=60) as response:
                data = response.read()
            target_jar.parent.mkdir(parents=True, exist_ok=True)
            target_jar.write_bytes(data)
            return True, "downloaded tla2tools.jar from network"
        except HTTPError as e:
            return False, f"HTTPError during TLC download: {e.code} {e.reason}"
        except URLError as e:
            return False, f"URLError during TLC download: {e.reason}"

    def _install_tlc_wrapper(self, jar_path: Path) -> None:
        wrapper = Path("/usr/local/bin/tlc")
        wrapper.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            f"exec java -cp {shlex.quote(str(jar_path))} tlc2.TLC \"$@\"\n"
        )
        wrapper.chmod(0o755)

    def _fallback_from_local_artifacts(self) -> Tuple[bool, str]:
        root = os.environ.get("LOCAL_ARTIFACT_ROOT")
        if not root:
            return False, "LOCAL_ARTIFACT_ROOT is not set"

        artifact_root = Path(root)
        if not artifact_root.exists():
            return False, f"LOCAL_ARTIFACT_ROOT does not exist: {artifact_root}"

        notes = []

        # TLC local artifact installation.
        local_tla_jar = artifact_root / "tla2tools.jar"
        if local_tla_jar.exists():
            target = Path("/usr/local/lib/tla2tools.jar")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local_tla_jar, target)
            self._install_tlc_wrapper(target)
            notes.append("installed TLC from LOCAL_ARTIFACT_ROOT/tla2tools.jar")
        else:
            notes.append("missing LOCAL_ARTIFACT_ROOT/tla2tools.jar")

        # Coq local artifact installation.
        # Supported options:
        # 1) LOCAL_ARTIFACT_ROOT/coq/bin/coqc (portable binary tree)
        # 2) LOCAL_ARTIFACT_ROOT/coqc
        # 3) LOCAL_ARTIFACT_ROOT/debs/*.deb (installed via dpkg -i)
        local_coq_bin = artifact_root / "coq" / "bin" / "coqc"
        local_flat_coqc = artifact_root / "coqc"
        deb_dir = artifact_root / "debs"

        if local_coq_bin.exists():
            target = Path("/usr/local/bin/coqc")
            shutil.copy2(local_coq_bin, target)
            target.chmod(0o755)
            notes.append("installed coqc from LOCAL_ARTIFACT_ROOT/coq/bin/coqc")
        elif local_flat_coqc.exists():
            target = Path("/usr/local/bin/coqc")
            shutil.copy2(local_flat_coqc, target)
            target.chmod(0o755)
            notes.append("installed coqc from LOCAL_ARTIFACT_ROOT/coqc")
        elif deb_dir.exists() and any(deb_dir.glob("*.deb")):
            debs = sorted(str(p) for p in deb_dir.glob("*.deb"))
            proc = self._run(["dpkg", "-i", *debs])
            notes.append(f"dpkg install attempted for {len(debs)} local .deb files")
            if proc.returncode != 0:
                notes.append("dpkg stderr: " + proc.stderr.strip())
        else:
            notes.append("missing Coq artifacts under LOCAL_ARTIFACT_ROOT")

        states = self.preflight()
        if self.all_tools_present(states):
            return True, "; ".join(notes)
        return False, "; ".join(notes)

    def install(self) -> int:
        states = self.preflight()
        if self.all_tools_present(states):
            self._record_attempt(
                "preflight",
                True,
                {k: vars(v) for k, v in states.items()},
            )
            print("All required tools already installed; skipping network/local installs.")
            return 0

        if self.should_skip_network():
            print("Skipping network install based on previous failed network attempt and no LOCAL_ARTIFACT_ROOT.")
            ok, note = self._fallback_from_local_artifacts()
            self._record_attempt("local_fallback", ok, {"note": note})
            print(note)
            return 0 if ok else 1

        ok, stdout, stderr = self.run_install_script()
        combined = (stdout or "") + "\n" + (stderr or "")
        self._record_attempt(
            "network_install",
            ok,
            {
                "stdout_tail": stdout[-2000:],
                "stderr_tail": stderr[-2000:],
            },
        )

        if ok:
            print(stdout, end="")
            print(stderr, end="", file=sys.stderr)
            post = self.preflight()
            self._record_attempt("post_network_preflight", self.all_tools_present(post), {k: vars(v) for k, v in post.items()})
            return 0 if self.all_tools_present(post) else 1

        status = self._http_status_from_error(combined)
        print(stdout, end="")
        print(stderr, end="", file=sys.stderr)

        if status in (403, 407):
            print(f"Detected HTTP {status} during network install.")
            print(self._suggest_proxy_flags())
            fallback_ok, note = self._fallback_from_local_artifacts()
            self._record_attempt("local_fallback", fallback_ok, {"note": note, "http_status": status})
            print(note)
            return 0 if fallback_ok else 1

        return 1

    def show_status(self) -> int:
        states = self.preflight()
        print(json.dumps({k: vars(v) for k, v in states.items()}, indent=2))
        return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Formal dependency manager for TLC and Coq.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="Run pre-flight checks and print tool status.")
    sub.add_parser("install", help="Install missing dependencies with network + local fallback strategy.")

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    engine = DepEngine(MANIFEST_PATH, INSTALL_SCRIPT)

    if args.cmd == "status":
        return engine.show_status()
    if args.cmd == "install":
        return engine.install()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
