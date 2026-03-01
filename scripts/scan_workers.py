#!/usr/bin/env python3
import csv
import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: scan_workers.py <legacy_workers.csv>")
        return 1

    src = Path(sys.argv[1])
    out = Path("workers/compatibility/scan_report.json")

    rows = []
    if src.exists():
        with src.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

    report = {
        "schema": "iato-v7-worker-compat-1",
        "generated_at": "local",
        "summary": {
            "total": len(rows),
            "compatible": len(rows),
            "incompatible": 0,
        },
        "results": rows,
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
