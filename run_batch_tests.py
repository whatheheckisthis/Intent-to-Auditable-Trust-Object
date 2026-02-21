#!/usr/bin/env python3
"""Offline bootstrap/build staging test automation framework.

This runner executes deterministic test batches for offline .NET bootstrap/build scripts,
collects wall-clock timing, and emits CSV logs for later notebook analysis.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import random
import shutil
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


FIXED_BATCHES = 4
FIXED_TESTS_PER_BATCH = 100


EXPECTED_EXIT_CODES = {
    "fail_fast_bootstrap": {42},
    "toolchain_presence_check": {0},
    "compile_only_build_attempt": {0},
    "missing_sdk": {42},
    "missing_staged_archive": {42},
    "invalid_architecture": {42, 43},
    "recovery_success": {0},
    "recovery_failure": {44},
}


@dataclass(frozen=True)
class Scenario:
    """Scenario contract.

    Each scenario models one offline bootstrap/build condition and returns a subprocess
    command + environment to execute. Timing is measured as wall-clock elapsed seconds
    around that subprocess invocation.
    """

    key: str
    description: str
    command_builder: Callable[[Path, random.Random, str], tuple[list[str], dict[str, str]]]


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _scripts_dir() -> Path:
    return _repo_root() / "scripts"


def _write_executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def _make_fake_dotnet_bin(bin_dir: Path) -> None:
    """Create a fake offline-compatible dotnet binary.

    Supports --info, --version, and build invocation used by compile-only checks.
    """

    dotnet_script = textwrap.dedent(
        """\
        #!/usr/bin/env bash
        set -euo pipefail
        if [[ "${1:-}" == "--info" ]]; then
          echo "Fake .NET SDK (offline)"
          exit 0
        fi
        if [[ "${1:-}" == "--version" ]]; then
          echo "9.0.100-offline"
          exit 0
        fi
        if [[ "${1:-}" == "build" ]]; then
          shift
          [[ "${1:-}" == "src/NfcReader/NfcReader.sln" ]] || { echo "unexpected solution" >&2; exit 2; }
          echo "Build succeeded (fake compile-only)."
          exit 0
        fi
        echo "unsupported dotnet args: $*" >&2
        exit 3
        """
    )
    _write_executable(bin_dir / "dotnet", dotnet_script)


def _make_fake_uname(bin_dir: Path, machine: str) -> None:
    _write_executable(
        bin_dir / "uname",
        f"#!/usr/bin/env bash\nif [[ \"${{1:-}}\" == \"-m\" ]]; then echo \"{machine}\"; else /usr/bin/uname \"$@\"; fi\n",
    )


def _make_fake_id_non_root(bin_dir: Path) -> None:
    _write_executable(
        bin_dir / "id",
        "#!/usr/bin/env bash\nif [[ \"${1:-}\" == \"-u\" ]]; then echo 1000; else /usr/bin/id \"$@\"; fi\n",
    )


def _create_fake_sdk_archive(archive_path: Path) -> None:
    """Create a staged SDK tar.gz with a runnable fake dotnet binary.

    This supports recovery-success flows without relying on network connectivity.
    """

    with tempfile.TemporaryDirectory(prefix="fake-sdk-root-") as temp_sdk:
        sdk_root = Path(temp_sdk)
        _make_fake_dotnet_bin(sdk_root)
        tools_dir = sdk_root / "tools"
        tools_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path, mode="w:gz") as tar:
            for child in sdk_root.iterdir():
                tar.add(child, arcname=child.name)


def _base_env(home_dir: Path, tme_seed: str, temp_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "HOME": str(home_dir),
            "TME_SIMULATION_MODE": "1",
            "TME_SEED": tme_seed,
            "PATH": f"{temp_path}:/usr/bin:/bin",
        }
    )
    return env


def _scenario_fail_fast_bootstrap(work_dir: Path, rng: random.Random, tme_seed: str) -> tuple[list[str], dict[str, str]]:
    """Fail-Fast SDK bootstrap.

    Runs ensure-dotnet without any staged archive and without dotnet on PATH;
    expected deterministic exit code is 42.
    """

    home = work_dir / "home"
    home.mkdir(parents=True, exist_ok=True)
    env = _base_env(home, tme_seed, work_dir / "bin")
    cmd = ["bash", str(_scripts_dir() / "ensure-dotnet.sh")]
    return cmd, env


def _scenario_toolchain_presence(work_dir: Path, rng: random.Random, tme_seed: str) -> tuple[list[str], dict[str, str]]:
    """Toolchain presence check.

    Installs a fake dotnet executable on PATH so ensure-dotnet validates immediate
    toolchain availability with a zero exit.
    """

    bin_dir = work_dir / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    _make_fake_dotnet_bin(bin_dir)
    home = work_dir / "home"
    home.mkdir(parents=True, exist_ok=True)
    env = _base_env(home, tme_seed, bin_dir)
    cmd = ["bash", str(_scripts_dir() / "ensure-dotnet.sh")]
    return cmd, env


def _scenario_compile_only(work_dir: Path, rng: random.Random, tme_seed: str) -> tuple[list[str], dict[str, str]]:
    """Compile-only build attempt (no restore/runtime).

    Uses fake dotnet to validate build script flags and command wiring while remaining
    fully offline and runtime-free.
    """

    bin_dir = work_dir / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    _make_fake_dotnet_bin(bin_dir)
    home = work_dir / "home"
    home.mkdir(parents=True, exist_ok=True)
    env = _base_env(home, tme_seed, bin_dir)

    dotnet_root = home / ".dotnet"
    dotnet_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(bin_dir / "dotnet", dotnet_root / "dotnet")
    (dotnet_root / "tools").mkdir(parents=True, exist_ok=True)

    cmd = ["bash", str(_scripts_dir() / "build-nfcreader-offline.sh")]
    return cmd, env


def _scenario_missing_sdk(work_dir: Path, rng: random.Random, tme_seed: str) -> tuple[list[str], dict[str, str]]:
    """Edge case: missing SDK (explicit missing archive path).

    Forces DOTNET_ARCHIVE to a non-existent path; ensure-dotnet should fail fast with
    code 42 after invoking recovery.
    """

    home = work_dir / "home"
    home.mkdir(parents=True, exist_ok=True)
    env = _base_env(home, tme_seed, work_dir / "bin")
    env["DOTNET_ARCHIVE"] = str(work_dir / "missing-dotnet-sdk.tar.gz")
    cmd = ["bash", str(_scripts_dir() / "ensure-dotnet.sh")]
    return cmd, env


def _scenario_missing_staged_archive(work_dir: Path, rng: random.Random, tme_seed: str) -> tuple[list[str], dict[str, str]]:
    """Edge case: missing staged archive for recovery script.

    Calls recover-dotnet directly with no bootstrap archives available.
    """

    home = work_dir / "home"
    home.mkdir(parents=True, exist_ok=True)
    env = _base_env(home, tme_seed, work_dir / "bin")
    cmd = ["bash", str(_scripts_dir() / "recover-dotnet-from-archive.sh")]
    return cmd, env


def _scenario_invalid_arch(work_dir: Path, rng: random.Random, tme_seed: str) -> tuple[list[str], dict[str, str]]:
    """Edge case: invalid architecture.

    Overrides uname -m to return unsupported machine type, expecting exit 43.
    """

    bin_dir = work_dir / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    _make_fake_uname(bin_dir, "mips64")
    home = work_dir / "home"
    home.mkdir(parents=True, exist_ok=True)
    env = _base_env(home, tme_seed, bin_dir)
    cmd = ["bash", str(_scripts_dir() / "recover-dotnet-from-archive.sh")]
    return cmd, env


def _scenario_recovery_success(work_dir: Path, rng: random.Random, tme_seed: str) -> tuple[list[str], dict[str, str]]:
    """Edge case: offline recovery success.

    Creates local staged SDK archive + non-root identity shim, then validates recovery
    can install and run dotnet with exit 0.
    """

    bin_dir = work_dir / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    _make_fake_id_non_root(bin_dir)
    _make_fake_uname(bin_dir, "x86_64")

    bootstrap_dir = work_dir / "bootstrap"
    bootstrap_dir.mkdir(parents=True, exist_ok=True)
    archive = bootstrap_dir / "dotnet-sdk-9.0.100-linux-x64.tar.gz"
    _create_fake_sdk_archive(archive)

    home = work_dir / "home"
    home.mkdir(parents=True, exist_ok=True)
    env = _base_env(home, tme_seed, bin_dir)
    env["DOTNET_ARCHIVE"] = str(archive)
    cmd = ["bash", str(_scripts_dir() / "recover-dotnet-from-archive.sh")]
    return cmd, env


def _scenario_recovery_failure(work_dir: Path, rng: random.Random, tme_seed: str) -> tuple[list[str], dict[str, str]]:
    """Edge case: recovery failure due to invalid archive.

    Produces a malformed tar.gz file to force extraction error, expecting exit 44.
    """

    bin_dir = work_dir / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    _make_fake_id_non_root(bin_dir)
    _make_fake_uname(bin_dir, "x86_64")

    archive = work_dir / "broken-sdk-linux-x64.tar.gz"
    archive.write_bytes(b"this-is-not-a-valid-tarball")

    home = work_dir / "home"
    home.mkdir(parents=True, exist_ok=True)
    env = _base_env(home, tme_seed, bin_dir)
    env["DOTNET_ARCHIVE"] = str(archive)
    cmd = ["bash", str(_scripts_dir() / "recover-dotnet-from-archive.sh")]
    return cmd, env


SCENARIOS = [
    Scenario("fail_fast_bootstrap", "Fail-Fast SDK bootstrap", _scenario_fail_fast_bootstrap),
    Scenario("toolchain_presence_check", "Toolchain presence check", _scenario_toolchain_presence),
    Scenario("compile_only_build_attempt", "Compile-only build attempt (no restore/no runtime)", _scenario_compile_only),
    Scenario("missing_sdk", "Edge: missing SDK", _scenario_missing_sdk),
    Scenario("missing_staged_archive", "Edge: missing staged archive", _scenario_missing_staged_archive),
    Scenario("invalid_architecture", "Edge: invalid architecture", _scenario_invalid_arch),
    Scenario("recovery_success", "Edge: recovery success", _scenario_recovery_success),
    Scenario("recovery_failure", "Edge: recovery failure", _scenario_recovery_failure),
]


CSV_COLUMNS = [
    "timestamp_utc",
    "batch_number",
    "test_id",
    "scenario_key",
    "scenario_description",
    "exit_code",
    "expected_exit_codes",
    "unexpected_exit_code",
    "execution_time_seconds",
    "warning_count",
    "error_count",
    "stdout_stderr",
    "command",
    "tme_seed",
]


def run_single_test(batch_number: int, test_id: int, scenario: Scenario, base_seed: int, output_dir: Path) -> dict[str, object]:
    """Execute one test case and return CSV row payload.

    Timing approach: wall-clock measurement via perf_counter surrounding the full
    subprocess execution for each scenario command.
    """

    seed_material = f"batch={batch_number}|test={test_id}|scenario={scenario.key}|seed={base_seed}"
    rng = random.Random(seed_material)
    tme_seed = f"tme-{rng.randrange(1, 10**12)}"

    with tempfile.TemporaryDirectory(prefix=f"batch{batch_number}_test{test_id}_", dir=output_dir) as temp_dir:
        work_dir = Path(temp_dir)
        cmd, env = scenario.command_builder(work_dir, rng, tme_seed)

        start = time.perf_counter()
        proc = subprocess.run(
            cmd,
            cwd=_repo_root(),
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        elapsed = time.perf_counter() - start

    merged_output = (proc.stdout or "") + (proc.stderr or "")
    warning_count = merged_output.lower().count("warning")
    error_count = merged_output.lower().count("error")

    expected_codes = EXPECTED_EXIT_CODES[scenario.key]
    unexpected = int(proc.returncode not in expected_codes)

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "batch_number": batch_number,
        "test_id": test_id,
        "scenario_key": scenario.key,
        "scenario_description": scenario.description,
        "exit_code": proc.returncode,
        "expected_exit_codes": "|".join(str(v) for v in sorted(expected_codes)),
        "unexpected_exit_code": unexpected,
        "execution_time_seconds": f"{elapsed:.6f}",
        "warning_count": warning_count,
        "error_count": error_count,
        "stdout_stderr": merged_output.strip().replace("\x00", ""),
        "command": " ".join(cmd),
        "tme_seed": tme_seed,
    }


def run_batches(total_batches: int, tests_per_batch: int, output_dir: Path, base_seed: int) -> None:
    """Run all batches and emit batch_<n>_results.csv files.

    CSV format includes scenario metadata, exit validation, timing, and captured logs.
    Deterministic behavior is anchored to base_seed and test identifiers.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    for batch in range(1, total_batches + 1):
        csv_path = output_dir / f"batch_{batch}_results.csv"
        rows: list[dict[str, object]] = []
        scenario_plan = build_fixed_batch_plan(batch, tests_per_batch, base_seed)

        for test_id, scenario in enumerate(scenario_plan, start=1):
            row = run_single_test(batch, test_id, scenario, base_seed, output_dir)
            rows.append(row)
            print(
                f"batch={batch} test={test_id} scenario={scenario.key} "
                f"exit={row['exit_code']} elapsed={row['execution_time_seconds']}s"
            )

        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline bootstrap/build staged test batches")
    parser.add_argument("--batches", type=int, default=FIXED_BATCHES, help=f"Number of batches (default: {FIXED_BATCHES})")
    parser.add_argument("--tests-per-batch", type=int, default=FIXED_TESTS_PER_BATCH, help=f"Tests per batch (default: {FIXED_TESTS_PER_BATCH})")
    parser.add_argument("--output-dir", type=Path, default=Path("."), help="Directory for batch CSV files")
    parser.add_argument("--seed", type=int, default=20260220, help="Deterministic base seed")
    parser.add_argument("--clean-existing", action="store_true", help="Delete existing batch_*_results.csv before running")
    return parser.parse_args()


def build_fixed_batch_plan(batch_number: int, tests_per_batch: int, base_seed: int) -> list[Scenario]:
    """Build a deterministic, fixed-count scenario plan for a batch.

    Scenario allocation uses a stable base distribution across all scenarios, then rotates
    remainder assignments by batch number to keep total counts fixed while balancing edge
    scenarios over 4x100 runs.
    """

    base = tests_per_batch // len(SCENARIOS)
    remainder = tests_per_batch % len(SCENARIOS)
    counts = {scenario.key: base for scenario in SCENARIOS}

    for i in range(remainder):
        idx = (batch_number - 1 + i) % len(SCENARIOS)
        counts[SCENARIOS[idx].key] += 1

    plan: list[Scenario] = []
    for scenario in SCENARIOS:
        plan.extend([scenario] * counts[scenario.key])

    random.Random(f"batch-plan|{base_seed}|{batch_number}").shuffle(plan)
    return plan


def ensure_optional_dependencies(auto_install: bool) -> None:
    """Ensure notebook analytics dependencies are available.

    Installation approach is offline-first for EL2/container workflows:
    - Attempt local wheelhouse installs only (`--no-index`) from `./vendor/wheels`
      or `OFFLINE_WHEELHOUSE` when dependencies are missing.
    - If no local wheelhouse exists, print a warning and continue without failing.
    """

    missing = [pkg for pkg in ("pandas", "matplotlib") if importlib.util.find_spec(pkg) is None]
    if not missing:
        return

    if not auto_install:
        print(f"warning: missing optional dependencies: {', '.join(missing)}")
        return

    wheelhouse = Path(os.environ.get("OFFLINE_WHEELHOUSE", _repo_root() / "vendor" / "wheels"))
    if not wheelhouse.exists():
        print(
            "warning: missing optional dependencies and no offline wheelhouse found "
            f"at {wheelhouse}; skipping auto-install"
        )
        return

    print(f"installing missing dependencies from offline wheelhouse: {', '.join(missing)}")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--find-links",
            str(wheelhouse),
            *missing,
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        print("warning: offline dependency auto-install failed; continuing without notebook dependencies")
        if proc.stderr:
            print(proc.stderr.strip())


def main() -> None:
    args = parse_args()
    if args.batches != FIXED_BATCHES or args.tests_per_batch != FIXED_TESTS_PER_BATCH:
        raise ValueError(
            f"strict fixed layout enabled: batches/tests must be {FIXED_BATCHES}/{FIXED_TESTS_PER_BATCH}, "
            f"got {args.batches}/{args.tests_per_batch}"
        )

    ensure_optional_dependencies(auto_install=True)

    if args.clean_existing:
        for csv_file in args.output_dir.glob("batch_*_results.csv"):
            csv_file.unlink()

    run_batches(args.batches, args.tests_per_batch, args.output_dir, args.seed)


if __name__ == "__main__":
    main()
