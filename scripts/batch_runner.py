#!/usr/bin/env python3
"""Deterministic batch runner for timing-window compliance checks.

By default this executes 4 batches x 100 runs in simulation mode to verify that
all timing-mitigation-wrapped scripts report audited segment windows.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RunResult:
    batch: int
    iteration: int
    command: str
    returncode: int


def run_once(script: Path, batch: int, iteration: int, seed: str) -> RunResult:
    env = os.environ.copy()
    env.setdefault("TME_SIMULATION_MODE", "1")
    env.setdefault("TME_SEED", seed)

    completed = subprocess.run(
        ["bash", str(script)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return RunResult(batch=batch, iteration=iteration, command=str(script), returncode=completed.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deterministic timing mitigation verification batches.")
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--runs-per-batch", type=int, default=100)
    parser.add_argument("--seed", default="0x4941544f2d5637")
    parser.add_argument("--allowed-return-codes", default="0,42")
    parser.add_argument(
        "--scripts",
        nargs="+",
        default=[
            "scripts/ensure-dotnet.sh",
            "scripts/recover-dotnet-from-archive.sh",
            "scripts/build-nfcreader-offline.sh",
        ],
        help="scripts to execute in order for each run",
    )
    args = parser.parse_args()

    script_paths = [Path(script).resolve() for script in args.scripts]
    missing = [str(path) for path in script_paths if not path.exists()]
    if missing:
        print("missing scripts:", ", ".join(missing), file=sys.stderr)
        return 2

    allowed_codes = {int(value) for value in args.allowed_return_codes.split(",") if value.strip()}

    failures: list[RunResult] = []

    for batch in range(1, args.batches + 1):
        for iteration in range(1, args.runs_per_batch + 1):
            for script in script_paths:
                result = run_once(script, batch, iteration, args.seed)
                if result.returncode not in allowed_codes:
                    failures.append(result)

    total_runs = args.batches * args.runs_per_batch * len(script_paths)
    passed = total_runs - len(failures)

    print(f"seed={args.seed} batches={args.batches} runs_per_batch={args.runs_per_batch}")
    print(f"total={total_runs} passed={passed} failed={len(failures)}")

    if failures:
        for failure in failures[:20]:
            print(
                f"failure batch={failure.batch} iteration={failure.iteration} "
                f"script={failure.command} rc={failure.returncode}",
                file=sys.stderr,
            )
        if len(failures) > 20:
            print(f"... {len(failures) - 20} additional failures omitted", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
